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

"""End-to-end TVCache demo with a scripted agent; no model or TPU needed.

Skipped unless TVCACHE_INTEGRATION=1. Needs a tvcache-rust server on port 7878
and a container manager on port 9901 with ENABLE_CACHING=1 and
TASKS_ROOT=../tasks_easy. Run with `pytest -s` to see the demo's output.
"""

import asyncio
import glob
import json
import os
import tempfile
import time
import uuid

from absl.testing import absltest
import numpy as np
from tunix.rl.agentic.agents import tool_agent
from tunix.rl.agentic.environments import terminal_bench_sandbox
from tunix.rl.agentic.environments import tvcache_env
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.rollout import base_rollout

_TASK = "hello-world"
_SOLVE = "echo 'Hello, world!' > /app/hello.txt"


def _tracking_sandbox(handles):
  """Returns a sandbox class recording handles, which outlive rollouts."""

  class _Sandbox(terminal_bench_sandbox.TerminalBenchSandbox):

    async def fork(self):
      handle = await super().fork()
      handles.append(handle.get_id())
      return handle

  return _Sandbox


async def _stop_all(handles):
  for handle in handles:
    await terminal_bench_sandbox.TerminalBenchSandbox(
        _TASK, env_id=handle
    ).stop()


@absltest.skipUnless(
    os.environ.get("TVCACHE_INTEGRATION") == "1",
    "Set TVCACHE_INTEGRATION=1 and start the servers to run.",
)
class TVCacheIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tempdir = tempfile.TemporaryDirectory()
    self.addCleanup(tempdir.cleanup)
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(tempdir.name)  # The executor logs to ./rollouts.
    # Unique per test, so reruns miss the cache. A shell comment would time out
    # by swallowing terminal-bench's completion signal.
    self.run_marker = f"true run-{uuid.uuid4().hex[:8]}"
    self.handles = []
    self.addCleanup(lambda: asyncio.run(_stop_all(self.handles)))
    self.pair_index = 0

  def _rollout(self, commands, label, executor_kwargs=None):
    """Runs one episode through the real engine and agent."""
    replies = [
        "<tool_call>"
        + json.dumps({"name": "terminal", "arguments": {"command": command}})
        + "</tool_call>"
        for command in [self.run_marker, *commands]
    ] + ["Done."]

    def model_call(*unused_args, **unused_kwargs):
      return base_rollout.RolloutOutput(
          text=[replies.pop(0)],
          logits=None,
          tokens=[np.array([])],
          left_padded_prompt_tokens=np.array([[]]),
          logprobs=None,
      )

    self.pair_index += 1
    env = tvcache_env.TVCacheEnv(
        {"task_name": _TASK, "question": "Create hello.txt."},
        sandbox_cls=_tracking_sandbox(self.handles),
        executor_kwargs=executor_kwargs,
        max_steps=len(commands) + 2,
        group_id=0,
        pair_index=self.pair_index,
    )
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        tool_agent.ToolAgent(system_prompt="Use the terminal."),
        env,
        model_call=model_call,
    )
    start = time.perf_counter()
    trajectory = asyncio.run(engine.collect(mode="Trajectory"))
    elapsed = time.perf_counter() - start

    messages = trajectory.steps[-1].chat_completions
    outputs = [m["content"] for m in messages if m["role"] == "tool"][1:]
    print(
        f"\n[{label}] {elapsed:.2f}s, reward {trajectory.reward},"
        f" {env.metrics['cache_hits']}/{env.metrics['tool_calls']} cache hits,"
        f" {env.metrics['executions']} executed"
    )
    for output in outputs:
      print(f"  {output}")
    return outputs, trajectory.reward, env.metrics, elapsed

  def test_identical_rollout_skips_execution(self):
    commands = ["ls", "echo hi > a.txt", "cat a.txt", _SOLVE]

    first, first_reward, first_metrics, first_time = self._rollout(
        commands, "rollout 1"
    )
    second, second_reward, second_metrics, second_time = self._rollout(
        commands, "rollout 2"
    )

    self.assertEqual(first_metrics["cache_hits"], 0)
    self.assertEqual(first_reward, 1.0)
    self.assertEqual(second, first)
    self.assertEqual(second_reward, first_reward)
    self.assertEqual(second_metrics["cache_hits"], second_metrics["tool_calls"])
    self.assertEqual(second_metrics["executions"], 0)
    self.assertLess(second_time, first_time)
    self.assertIn("root@terminal:/app# cat a.txt\nhi", "\n".join(first))

    logs = sorted(glob.glob("rollouts/*.log"), key=os.path.getmtime)
    with open(logs[-1]) as f:
      second_log = f.read()
    self.assertIn("CACHE HIT", second_log)
    self.assertNotIn("CACHE MISS", second_log)
    self.assertNotIn("Started fresh environment", second_log)

  def test_divergent_rollout_reuses_prefix_and_replays_state(self):
    shared = ["cd /tmp", "echo hi > a.txt"]
    self._rollout([*shared, "cat a.txt"], "original")
    outputs, reward, metrics, _ = self._rollout(
        [*shared, "cat a.txt; pwd"], "divergent"
    )
    self.assertIn("hi\n/tmp", outputs[-1])
    self.assertEqual(reward, 0.0)
    self.assertEqual(metrics["cache_hits"], 3)  # Marker and 2 shared commands.
    # Replayed marker and 2 shared, plus the new command and the test.
    self.assertEqual(metrics["executions"], 5)

  def test_resuming_from_snapshot_keeps_state_and_isolation(self):
    snapshots = {"adaptive": False, "fixed_threshold_seconds": 1e-9}
    shared = ["cd /tmp", "export VALUE=kept", "echo hi > /app/a.txt"]
    self._rollout([*shared, "true"], "snapshotting", snapshots)
    self.assertNotEmpty(self.handles)

    probe = "pwd; echo VALUE=$VALUE; cat /app/a.txt; ls /app"
    resumed, _, first_metrics, _ = self._rollout(
        [*shared, "echo first > /app/only.txt", probe], "resume 1", snapshots
    )
    # The second resume must not see the first one's file.
    other, _, second_metrics, _ = self._rollout(
        [*shared, probe], "resume 2", snapshots
    )

    self.assertEqual(first_metrics["cache_hits"], 4)  # Marker and 3 shared.
    self.assertEqual(second_metrics["cache_hits"], 4)
    self.assertIn("/tmp\nVALUE=kept\nhi", resumed[-1])
    self.assertIn("only.txt", resumed[-1])
    self.assertIn("/tmp\nVALUE=kept\nhi", other[-1])
    self.assertNotIn("only.txt", other[-1])


if __name__ == "__main__":
  absltest.main()
