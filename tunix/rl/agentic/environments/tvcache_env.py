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

"""Terminal environment whose tool calls go through a TVCache server.

Requires the optional `tvclient` and `httpx` packages.
"""

import asyncio
import concurrent.futures
import functools
import json
import logging as std_logging
import threading
import time
from typing import Any, Coroutine, Dict, List, Optional, Type
import uuid

from absl import logging
import numpy as np
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments import terminal_bench_sandbox
from tvclient import executor as tvclient_executor
from tvclient import sandbox as tvclient_sandbox

ShellToolCall = terminal_bench_sandbox.ShellToolCall

# Must fit in the 150 s the trajectory engine gives `env.close()`.
_CLOSE_TIMEOUT_SECONDS = 120.0
_LOOP_JOIN_TIMEOUT_SECONDS = 5.0


class _BackgroundLoop:
  """Runs coroutines on one event loop from synchronous worker threads.

  `tvclient`'s connection is bound to the loop that opened it, so all calls for
  an env must share a loop.
  """

  def __init__(self):
    self._loop = asyncio.new_event_loop()
    self._lock = threading.Lock()
    self._inflight: set[concurrent.futures.Future] = set()
    self._closed = False
    self._thread = threading.Thread(
        target=self._loop.run_forever, name="tvcache-env-loop", daemon=True
    )
    self._thread.start()

  def run(
      self, coro: Coroutine[Any, Any, Any], timeout: Optional[float]
  ) -> Any:
    with self._lock:
      if self._closed:
        coro.close()
        raise RuntimeError("TVCacheEnv was closed while a step was running.")
      future = asyncio.run_coroutine_threadsafe(coro, self._loop)
      self._inflight.add(future)
    try:
      return future.result(timeout)
    except TimeoutError:
      future.cancel()
      raise
    finally:
      with self._lock:
        self._inflight.discard(future)

  def cancel_inflight(self) -> None:
    with self._lock:
      for future in self._inflight:
        future.cancel()

  def stop(self) -> None:
    with self._lock:
      self._closed = True
    self.cancel_inflight()
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._thread.join(timeout=_LOOP_JOIN_TIMEOUT_SECONDS)
    if self._thread.is_alive():
      logging.warning("TVCache event loop thread did not stop in time.")
    else:
      self._loop.close()


def _unwrap_scalar(value: Any) -> Any:
  """Unwraps the 1-element arrays that learner batches leave in task fields."""
  if isinstance(value, np.ndarray) and value.size == 1:
    value = value.item()
  elif isinstance(value, (list, tuple)) and len(value) == 1:
    value = value[0]
  if isinstance(value, bytes):
    value = value.decode()
  return value


class TVCacheEnv(base_environment.BaseTaskEnv):
  """Runs an agent's terminal commands through TVCache.

  Tool calls carry a `command` argument, and observations match
  `ToolEnvironment`. The reward comes from the task's tests, through the
  engine's `final_reward_fn` hook, so it also covers episodes that end at
  `max_steps`.
  """

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      *,
      tvcache_host: str = "127.0.0.1",
      tvcache_port: int = 7878,
      sandbox_cls: Type[
          tvclient_sandbox.Sandbox
      ] = terminal_bench_sandbox.TerminalBenchSandbox,
      sandbox_kwargs: Dict[str, Any] | None = None,
      executor_kwargs: Dict[str, Any] | None = None,
      task_name_key: str = "task_name",
      tool_call_timeout: float | None = 3600.0,
      max_steps: int = 10,
      **kwargs,
  ):
    """Initializes the environment. No connection is made until `reset`.

    Args:
      task: Must hold the task's name under `task_name_key`, and normally its
        instruction under `question`.
      tvcache_host: Host of the TVCache server.
      tvcache_port: Port of the TVCache server.
      sandbox_cls: `tvclient.Sandbox` that runs commands on a cache miss.
      sandbox_kwargs: Extra constructor arguments for `sandbox_cls`.
      executor_kwargs: Extra arguments for `TvcacheToolExecutor`.
      task_name_key: Key of the task's name in `task`.
      tool_call_timeout: Limit in seconds for one call into the cache.
      max_steps: Maximum number of interaction steps.
      **kwargs: Stored by `BaseTaskEnv`, for example `group_id`.
    """
    super().__init__(task=task, max_steps=max_steps, **kwargs)
    self._tvcache_host = tvcache_host
    self._tvcache_port = tvcache_port
    self._sandbox_factory = functools.partial(
        sandbox_cls, **(sandbox_kwargs or {})
    )
    self._executor_kwargs = executor_kwargs or {}
    self._task_name_key = task_name_key
    self._tool_call_timeout = tool_call_timeout
    self._loop: Optional[_BackgroundLoop] = None
    self._executor: Optional[tvclient_executor.TvcacheToolExecutor] = None
    self._history: List[ShellToolCall] = []
    self.metrics = self._empty_metrics()

  @staticmethod
  def _empty_metrics() -> Dict[str, float]:
    return {
        # Calls into the cache, including the final test.
        "tool_calls": 0,
        # Calls that ran nothing.
        "cache_hits": 0,
        # Commands really run, including replays while resuming.
        "executions": 0,
        "tool_wait_seconds": 0.0,
    }

  def _rollout_id(self, task_name: str) -> str:
    group_id = self.extra_kwargs.get("group_id")
    pair_index = self.extra_kwargs.get("pair_index")
    return f"{task_name}-g{group_id}-p{pair_index}-{uuid.uuid4().hex[:8]}"

  def _initial_observation(self) -> Dict[str, Any]:
    """Connects to the cache for a new episode and returns the task."""
    self._shutdown()
    task_name = _unwrap_scalar(self.task.get(self._task_name_key))
    if not isinstance(task_name, str) or not task_name:
      raise ValueError(
          f"Task needs a task name under {self._task_name_key!r}, got"
          f" {task_name!r}."
      )
    self._history = []
    self.metrics = self._empty_metrics()
    self._loop = _BackgroundLoop()
    self._executor = tvclient_executor.TvcacheToolExecutor(
        self._sandbox_factory,  # pyrefly: ignore[bad-argument-type]
        ShellToolCall,
        task_id=task_name,
        host=self._tvcache_host,
        port=self._tvcache_port,
        **self._executor_kwargs,
    )
    try:
      self._loop.run(self._executor.client.connect(), self._tool_call_timeout)
    except BaseException:
      self._shutdown()
      raise
    self._executor.set_rollout_id(self._rollout_id(task_name))
    # The executor logs to its own per-rollout file.
    self._executor.logger.propagate = False
    return self.task

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    """Runs the action's commands in order; a `finish` ends the episode."""
    if isinstance(action, agent_types.Action):
      action = action.action
    if action is None:
      action = []
    if isinstance(action, dict):
      action = [action]

    done = isinstance(action, str) or any(
        call.get("function", {}).get("name") == "finish" for call in action
    )
    if done:
      return base_environment.EnvStepResult(
          observation={}, reward=0.0, done=True, info={"response": action}
      )

    # In order, never in parallel: the cache key is an ordered chain.
    outputs = {}
    for call in action:
      call_id = call.get("id") or str(uuid.uuid4())
      outputs[call_id] = self._run_tool_call(call)
    return base_environment.EnvStepResult(
        observation={"tool_outputs": outputs},
        reward=0.0,
        done=False,
        info={"response": action, "tvcache": dict(self.metrics)},
    )

  def _run_tool_call(self, call: Dict[str, Any]) -> str:
    """Returns a command's output, or an error the agent can react to."""
    function = call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
      try:
        arguments = json.loads(arguments)
      except json.JSONDecodeError as error:
        return f"Error: invalid arguments for {name}: {error}"
    command = arguments.get("command") if isinstance(arguments, dict) else None
    if not isinstance(command, str) or not command.strip():
      return f"Error: {name} needs a non-empty string argument 'command'."

    self._history.append(ShellToolCall(command, name=name))
    return self._call_cache(self._executor.lookup_or_execute(self._history))

  def _call_cache(self, coro: Coroutine[Any, Any, str]) -> str:
    executions_before = self._executor.total_executions
    start = time.perf_counter()
    try:
      return self._loop.run(coro, self._tool_call_timeout)
    finally:
      executed = self._executor.total_executions - executions_before
      self.metrics["tool_calls"] += 1
      self.metrics["cache_hits"] += int(executed == 0)
      self.metrics["executions"] += executed
      self.metrics["tool_wait_seconds"] += time.perf_counter() - start

  def final_reward_fn(self) -> float:
    """Runs the task's tests, or reuses their cached result."""
    result = self._call_cache(self._executor.test(list(self._history)))
    if result == terminal_bench_sandbox.PASSED:
      return 1.0
    if result != terminal_bench_sandbox.FAILED:
      logging.warning("TVCache test call did not complete: %s", result)
    return 0.0

  def close(self) -> None:
    self._shutdown()

  def _shutdown(self) -> None:
    loop, executor = self._loop, self._executor
    self._loop, self._executor = None, None
    if loop is None:
      return
    # A step the engine timed out may still be running in its worker thread.
    loop.cancel_inflight()
    try:
      if executor is not None:
        loop.run(executor.close(), _CLOSE_TIMEOUT_SECONDS)
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to close TVCache executor: %r", error)
    finally:
      loop.stop()
      if executor is not None:
        _close_log_handlers(executor.logger)


def _close_log_handlers(logger: std_logging.Logger) -> None:
  """Closes the file handler `set_rollout_id` adds, so it does not leak."""
  for handler in list(logger.handlers):
    logger.removeHandler(handler)
    handler.close()
