# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic tests: the real executor over an in-memory cache and fake server."""

import asyncio
import collections
import json
import os
import tempfile
import threading
import time
from unittest import mock
import uuid

from absl.testing import absltest
import httpx
import numpy as np
from tunix.rl.agentic.agents import tool_agent
from tunix.rl.agentic.environments import terminal_bench_sandbox
from tunix.rl.agentic.environments import tvcache_env
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.rollout import base_rollout
from tvclient import protocol
from tvclient import tool_call as tvclient_tool_call

_PROMPT = "root@abc123:/app#"
_SUFFIX = "; tmux wait -S done"


class _Found:
  """Stands in for `tvclient.FindResult`."""

  def __init__(self, value, sandbox_path, sandbox_id):
    self.value = value
    self.sandbox_path = sandbox_path
    self.sandbox_id = sandbox_id

  async def __aenter__(self):
    return self

  async def __aexit__(self, *args):
    return None


class _FakeCache:
  """In-memory longest-prefix cache shared by all clients of a test."""

  def __init__(self):
    self.values = {}
    self.sandboxes = {}

  def client(self, *unused_args, **unused_kwargs):
    return _FakeCacheClient(self)


class _FakeCacheClient:
  """Implements the part of `TVCacheClient` the executor uses."""

  def __init__(self, cache):
    self._cache = cache

  async def connect(self):
    return None

  async def close(self):
    return None

  async def find(self, task_id, path):
    value = self._cache.values.get((task_id, tuple(path)))
    if value is not None:
      return _Found(value, None, None)
    for depth in range(len(path) - 1, 0, -1):
      sandbox = self._cache.sandboxes.get((task_id, tuple(path[:depth])))
      if sandbox is not None:
        return _Found(None, list(path[:depth]), sandbox)
    return _Found(None, None, None)

  async def insert(self, task_id, path, value, sandbox_id, **unused_kwargs):
    self._cache.values[(task_id, tuple(path))] = value
    if sandbox_id is not None:
      self._cache.sandboxes[(task_id, tuple(path))] = sandbox_id
    return protocol.InsertResult(True, sandbox_id is not None)

  async def get_fork_cost(self, unused_task_id):
    return 0

  async def report_fork_cost(self, unused_task_id, fork_cost_ms):
    return fork_cost_ms


class _FakeContainerManager:
  """Implements the container manager's API over `httpx.MockTransport`."""

  def __init__(self):
    self.containers = {}
    self.calls = collections.Counter()
    self.execute_status = 200
    self.test_status = 200

  def _new_container(self, commands, pane):
    terminal_id = f"term_{uuid.uuid4().hex[:8]}"
    self.containers[terminal_id] = {"commands": commands, "pane": pane}
    return terminal_id

  def __call__(self, request: httpx.Request) -> httpx.Response:
    path = request.url.path
    body = json.loads(request.content)
    self.calls[path] += 1
    if path == "/start":
      return httpx.Response(
          200, json={"id": self._new_container([], [_PROMPT])}
      )

    container = self.containers.get(body["id"])
    if container is None:
      return httpx.Response(404, json={"error": "Session not found"})
    if path == "/fork":
      new_id = self._new_container(
          list(container["commands"]), list(container["pane"])
      )
      return httpx.Response(200, json={"id": new_id})
    if path == "/execute":
      if self.execute_status != 200:
        return httpx.Response(self.execute_status, json={"error": "timed out"})
      command = body["command"]
      # The output depends on state, so running in a wrong container shows.
      container["commands"].append(command)
      output = f"{command} ran as number {len(container['commands'])}"
      container["pane"][-1] += f" {command}{_SUFFIX}"
      container["pane"] += [output, _PROMPT]
      return httpx.Response(200, json={"status": "executed"})
    if path == "/get_pane_logs":
      return httpx.Response(200, json={"logs": "\n".join(container["pane"])})
    if path == "/test":
      if self.test_status != 200:
        return httpx.Response(self.test_status, json={"error": "broken"})
      return httpx.Response(
          200, json={"passed": "solve" in container["commands"]}
      )
    if path == "/stop":
      del self.containers[body["id"]]
      return httpx.Response(200, json={"status": "stopped"})
    return httpx.Response(404)


def _sandbox_class(manager):
  """Returns a `TerminalBenchSandbox` that talks to `manager`."""

  class _Sandbox(terminal_bench_sandbox.TerminalBenchSandbox):

    def __init__(self, task_name, env_id=None, **kwargs):
      super().__init__(task_name, env_id, **kwargs)
      self._client = httpx.AsyncClient(
          base_url=self.base_url, transport=httpx.MockTransport(manager)
      )

  return _Sandbox


def _call(command, call_id="call"):
  return {
      "id": call_id,
      "function": {
          "name": "terminal",
          "arguments": json.dumps({"command": command}),
      },
  }


class ExtractNewOutputTest(absltest.TestCase):

  def test_returns_echo_and_output_without_prompt_or_tmux_suffix(self):
    before = "root@abc:/app# ls\nfile\nroot@abc:/app#"
    after = (
        "root@abc:/app# ls\nfile\n"
        f"root@abc:/app# cat a{_SUFFIX}\nhi\nroot@abc:/app#"
    )
    self.assertEqual(
        terminal_bench_sandbox.extract_new_output(before, after),
        "root@terminal:/app# cat a\nhi",
    )

  def test_missing_trailing_prompt_gives_the_same_result(self):
    before = "root@abc:/app#"
    with_prompt = f"root@abc:/app# ls{_SUFFIX}\nfile\nroot@abc:/app#"
    without_prompt = f"root@abc:/app# ls{_SUFFIX}\nfile"
    self.assertEqual(
        terminal_bench_sandbox.extract_new_output(before, with_prompt),
        terminal_bench_sandbox.extract_new_output(before, without_prompt),
    )

  def test_hostname_of_non_root_user_is_normalized(self):
    self.assertEqual(
        terminal_bench_sandbox.extract_new_output(
            "alice@abc:~$", f"alice@abc:~$ ls{_SUFFIX}\nalice@def:~$"
        ),
        "alice@terminal:~$ ls",
    )

  def test_rewritten_pane_falls_back_to_last_command_echo(self):
    before = "line1\nline2\nline3\nroot@abc:/app#"
    after = (
        f"line3\nroot@abc:/app# a{_SUFFIX}\nx\n"
        f"root@abc:/app# b{_SUFFIX}\ny\nroot@abc:/app#"
    )
    self.assertEqual(
        terminal_bench_sandbox.extract_new_output(before, after),
        "root@terminal:/app# b\ny",
    )


class TerminalBenchSandboxTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.manager = _FakeContainerManager()
    self.sandbox_cls = _sandbox_class(self.manager)

  def _started(self):
    sandbox = self.sandbox_cls("task")
    asyncio.run(sandbox.start())
    return sandbox

  def test_start_on_handle_forks_and_leaves_handle_untouched(self):
    original = self._started()
    handle_id = asyncio.run(original.fork()).get_id()

    restored = self.sandbox_cls("task", env_id=handle_id)
    asyncio.run(restored.start())

    self.assertNotEqual(restored.get_id(), handle_id)
    self.assertIn(handle_id, self.manager.containers)
    self.assertEqual(self.manager.calls["/start"], 1)

  def test_execute_returns_only_new_output(self):
    sandbox = self._started()
    call = terminal_bench_sandbox.ShellToolCall
    first = asyncio.run(sandbox.execute(call("a")))
    second = asyncio.run(sandbox.execute(call("b")))
    self.assertEqual(first, "root@terminal:/app# a\na ran as number 1")
    self.assertEqual(second, "root@terminal:/app# b\nb ran as number 2")

  def test_server_errors_become_uncacheable_failures(self):
    sandbox = self._started()
    self.manager.execute_status = 504
    value = asyncio.run(
        sandbox.execute(terminal_bench_sandbox.ShellToolCall("sleep 999"))
    )
    self.assertStartsWith(value, terminal_bench_sandbox.FAILURE_PREFIX)
    self.manager.test_status = 404
    value = asyncio.run(sandbox.execute(tvclient_tool_call.TestToolCall()))
    self.assertStartsWith(value, terminal_bench_sandbox.FAILURE_PREFIX)

  def test_test_call_maps_to_canonical_strings(self):
    sandbox = self._started()
    test_call = tvclient_tool_call.TestToolCall()
    self.assertEqual(
        asyncio.run(sandbox.execute(test_call)), terminal_bench_sandbox.FAILED
    )
    asyncio.run(sandbox.execute(terminal_bench_sandbox.ShellToolCall("solve")))
    self.assertEqual(
        asyncio.run(sandbox.execute(test_call)), terminal_bench_sandbox.PASSED
    )


class TVCacheEnvTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(tempdir.cleanup)
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(tempdir.name)  # The executor logs to ./rollouts.

    self.cache = _FakeCache()
    patcher = mock.patch(
        "tvclient.executor.TVCacheClient", side_effect=self.cache.client
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    self.manager = _FakeContainerManager()

  def _env(self, **kwargs):
    kwargs.setdefault("task", {"task_name": "task", "question": "Do it."})
    kwargs.setdefault("sandbox_cls", _sandbox_class(self.manager))
    return tvcache_env.TVCacheEnv(**kwargs)

  def _rollout(self, commands, **kwargs):
    """Runs one episode directly and returns (outputs, reward, metrics)."""
    env = self._env(**kwargs)
    env.reset()
    outputs = []
    for command in commands:
      obs, _, _, _ = env.step([_call(command)])
      outputs.append(obs["tool_outputs"]["call"])
    env.step([{"function": {"name": "finish", "arguments": {"response": ""}}}])
    reward = env.final_reward_fn()
    metrics = dict(env.metrics)
    env.close()
    return outputs, reward, metrics

  def test_second_rollout_hits_cache_and_starts_nothing(self):
    commands = ["ls", "echo hi > a.txt", "solve"]

    first, first_reward, first_metrics = self._rollout(commands)
    self.assertEqual(self.manager.calls["/start"], 1)
    self.assertEqual(first_metrics["cache_hits"], 0)
    self.assertEqual(first_metrics["executions"], 4)  # 3 commands + test.

    self.manager.calls.clear()
    second, second_reward, second_metrics = self._rollout(commands)

    self.assertEqual(second, first)
    self.assertEqual(second_reward, first_reward)
    self.assertEqual(second_reward, 1.0)
    self.assertEqual(second_metrics["cache_hits"], 4)
    self.assertEqual(second_metrics["executions"], 0)
    self.assertEmpty(self.manager.calls)
    self.assertEmpty(self.manager.containers)

  def test_same_command_after_different_history_misses(self):
    self._rollout(["a", "c"])
    outputs, _, metrics = self._rollout(["b", "c"])
    self.assertEqual(outputs[1], "root@terminal:/app# c\nc ran as number 2")
    self.assertEqual(metrics["cache_hits"], 0)

  def test_divergence_without_snapshot_replays_prefix(self):
    self._rollout(["a", "b"])
    outputs, _, metrics = self._rollout(["a", "c"])
    self.assertEqual(outputs[1], "root@terminal:/app# c\nc ran as number 2")
    self.assertEqual(metrics["cache_hits"], 1)  # "a".
    self.assertEqual(metrics["executions"], 3)  # Replayed "a", "c", test.

  def test_many_resumes_from_one_snapshot_stay_correct(self):
    snapshots = {"adaptive": False, "fixed_threshold_seconds": 1e-9}
    self._rollout(["a", "b"], executor_kwargs=snapshots)
    handles = dict(self.cache.sandboxes)
    self.assertNotEmpty(handles)

    # Running in a handle's own container would break the next resume.
    for command in ["c", "d"]:
      self.manager.calls.clear()
      outputs, _, _ = self._rollout(["a", command], executor_kwargs=snapshots)
      self.assertEqual(self.manager.calls["/start"], 0)
      self.assertEqual(
          outputs[1],
          f"root@terminal:/app# {command}\n{command} ran as number 2",
      )

    for handle in handles.values():
      self.assertIn(handle, self.manager.containers)

  def test_task_name_is_unwrapped_from_batch_array(self):
    self._rollout(["a"], task={"task_name": np.array(["hello-world"])})
    self.assertTrue(all(key[0] == "hello-world" for key in self.cache.values))

  def test_missing_task_name_raises(self):
    with self.assertRaisesRegex(ValueError, "task name"):
      self._env(task={}).reset()

  def test_bad_arguments_return_error_and_do_not_extend_history(self):
    env = self._env()
    env.reset()
    obs, _, done, _ = env.step(
        [{"id": "x", "function": {"name": "terminal", "arguments": "{oops"}}]
    )
    self.assertStartsWith(obs["tool_outputs"]["x"], "Error:")
    self.assertFalse(done)
    obs, _, _, _ = env.step(
        [{"id": "y", "function": {"name": "terminal", "arguments": "{}"}}]
    )
    self.assertStartsWith(obs["tool_outputs"]["y"], "Error:")
    self.assertEqual(env.metrics["tool_calls"], 0)
    env.close()

  def test_failed_test_call_gives_zero_reward_and_is_not_cached(self):
    self.manager.test_status = 404
    _, reward, _ = self._rollout(["solve"])
    self.assertEqual(reward, 0.0)
    self.manager.test_status = 200
    _, reward, _ = self._rollout(["solve"])
    self.assertEqual(reward, 1.0)

  def test_close_stops_loop_thread_and_log_handlers(self):
    threads_before = threading.active_count()
    env = self._env()
    env.reset()
    logger = env._executor.logger  # pylint: disable=protected-access
    self.assertNotEmpty(logger.handlers)
    env.close()
    env.close()
    self.assertEmpty(logger.handlers)
    self.assertEqual(threading.active_count(), threads_before)

  def test_close_cancels_a_hung_step_and_stops_its_sandbox(self):
    entered = threading.Event()
    stopped = []

    class _HangingSandbox(_sandbox_class(self.manager)):

      async def execute(self, tool_call):
        entered.set()
        await asyncio.Event().wait()

      async def stop(self):
        stopped.append(self.get_id())
        await super().stop()

    env = self._env(sandbox_cls=_HangingSandbox)
    env.reset()
    errors = []

    def _step():
      try:
        env.step([_call("hang")])
      except BaseException as error:  # pylint: disable=broad-exception-caught
        errors.append(error)

    worker = threading.Thread(target=_step)
    worker.start()
    self.assertTrue(entered.wait(timeout=10))
    start = time.monotonic()
    env.close()
    worker.join(timeout=10)

    self.assertFalse(worker.is_alive())
    self.assertLen(errors, 1)
    self.assertLess(time.monotonic() - start, 10)
    self.assertLen(stopped, 1)
    self.assertEmpty(self.manager.containers)


class TVCacheEnvEngineTest(absltest.TestCase):
  """Drives the env through the real engine and agent, with a scripted model."""

  def setUp(self):
    super().setUp()
    tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(tempdir.cleanup)
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(tempdir.name)
    self.cache = _FakeCache()
    patcher = mock.patch(
        "tvclient.executor.TVCacheClient", side_effect=self.cache.client
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    self.manager = _FakeContainerManager()

  def _collect(self, replies, max_steps=10):
    remaining = list(replies)

    def model_call(*unused_args, **unused_kwargs):
      text = remaining.pop(0)
      return base_rollout.RolloutOutput(
          text=[text],
          logits=None,
          tokens=[np.array([])],
          left_padded_prompt_tokens=np.array([[]]),
          logprobs=None,
      )

    env = tvcache_env.TVCacheEnv(
        {"task_name": "task", "question": "Do it."},
        sandbox_cls=_sandbox_class(self.manager),
        max_steps=max_steps,
        group_id=0,
        pair_index=0,
    )
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        tool_agent.ToolAgent(system_prompt="sys"),
        env,
        model_call=model_call,
    )
    trajectory = asyncio.run(engine.collect(mode="Trajectory"))
    return trajectory, env

  @staticmethod
  def _reply(command):
    tool_call = {"name": "terminal", "arguments": {"command": command}}
    return f"<tool_call>{json.dumps(tool_call)}</tool_call>"

  def test_finish_earns_reward_and_second_episode_runs_nothing(self):
    replies = [self._reply("solve"), "All done."]

    first, _ = self._collect(replies)
    self.assertEqual(first.reward, 1.0)

    self.manager.calls.clear()
    second, env = self._collect(replies)
    self.assertEqual(second.reward, 1.0)
    self.assertEqual(env.metrics["executions"], 0)
    self.assertEmpty(self.manager.calls)
    self.assertEqual(
        [m for m in first.steps[-1].chat_completions if m["role"] == "tool"][0][
            "content"
        ],
        [m for m in second.steps[-1].chat_completions if m["role"] == "tool"][
            0
        ]["content"],
    )

  def test_reaching_max_steps_still_runs_the_tests(self):
    trajectory, _ = self._collect([self._reply("solve")], max_steps=1)
    self.assertEqual(trajectory.reward, 1.0)


if __name__ == "__main__":
  absltest.main()
