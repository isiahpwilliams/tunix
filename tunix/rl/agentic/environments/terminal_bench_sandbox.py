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

"""TVCache sandbox backed by the terminal-bench container manager.

Requires the optional `tvclient` and `httpx` packages.
"""

import asyncio
import re
import time
from typing import Any, Optional

import httpx
from tvclient import sandbox as tvclient_sandbox
from tvclient import tool_call as tvclient_tool_call

DEFAULT_CONTAINER_MANAGER_URL = "http://127.0.0.1:9901"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180.0
# Booting a task can include a Docker image build.
_START_TIMEOUT_SECONDS = 1800.0
_FORK_TIMEOUT_SECONDS = 600.0
_TEST_TIMEOUT_SECONDS = 300.0
_REQUEST_TIMEOUT_SECONDS = 60.0
_COMMAND_TIMEOUT_MARGIN_SECONDS = 30.0

PASSED = "PASSED"
FAILED = "FAILED"
# `TvcacheToolExecutor` never caches values with this prefix.
FAILURE_PREFIX = "Failed to execute"

# Appended to blocking commands by terminal-bench and echoed by the shell.
_TMUX_COMPLETION_SUFFIX = "; tmux wait -S done"
_USER_AT_HOST = re.compile(r"^([^\s@]+)@[^\s:]+:")
_BARE_PROMPT = re.compile(r"^[^\s@]+@[^\s:]+:.*[#$]\s*$")
# `tmux wait -S done` can fire before the output and prompt are rendered.
_SETTLE_POLL_SECONDS = 0.05
_SETTLE_TIMEOUT_SECONDS = 2.0


class ShellToolCall(tvclient_tool_call.ToolCall):
  """A terminal command; one node of the cache key."""

  def __init__(self, command: str, name: str = "terminal"):
    self.name = name
    self.command = command

  def to_dict(self) -> dict[str, str]:
    return {"name": self.name, "command": self.command}

  def will_mutate_state(self) -> bool:
    # Any shell command can change state; a wrong False causes bad cache hits.
    return True


def normalize_pane(pane: str) -> list[str]:
  """Returns pane lines without artifacts that differ between identical runs."""
  lines = []
  for line in pane.splitlines():
    line = _USER_AT_HOST.sub(r"\1@terminal:", line.rstrip())
    lines.append(line.replace(_TMUX_COMPLETION_SUFFIX, ""))
  return lines


def extract_new_output(before: str, after: str) -> str:
  """Returns the command echo and output that a command added to a tmux pane.

  The trailing prompt is dropped because its presence depends on timing.
  """
  before_lines = normalize_pane(before)
  after_lines = normalize_pane(after)
  start = 0
  for old, new in zip(before_lines, after_lines):
    if old != new:
      break
    start += 1

  # More than the last prompt line changed, so the pane was rewritten (`clear`
  # or scrollback overflow). Fall back to the last command echo.
  if start < len(before_lines) - 1:
    raw_lines = after.splitlines()
    start = 0
    for index in range(len(raw_lines) - 1, -1, -1):
      if _TMUX_COMPLETION_SUFFIX in raw_lines[index]:
        start = index
        break

  new_lines = after_lines[start:]
  if new_lines and _BARE_PROMPT.match(new_lines[-1]):
    new_lines = new_lines[:-1]
  return "\n".join(new_lines)


class TerminalBenchSandbox(tvclient_sandbox.Sandbox):
  """`tvclient.Sandbox` over the container manager's HTTP API.

  `start()` on a cached handle forks it, so the handle's own container is never
  executed in and many rollouts can resume from it.

  TODO(team34): `/fork` starts a container, while `tvclient` expects a cold
  snapshot, so every cached handle is a running container. This is the likely
  cause of "exploding environments".
  """

  def __init__(
      self,
      task_name: str,
      env_id: Optional[str] = None,
      *,
      base_url: str = DEFAULT_CONTAINER_MANAGER_URL,
      command_timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
  ):
    super().__init__(task_name, env_id)
    self.base_url = base_url
    self.command_timeout = command_timeout
    self._client: Optional[httpx.AsyncClient] = None

  async def _post(
      self, path: str, payload: dict[str, Any], timeout: float
  ) -> httpx.Response:
    if self._client is None:
      self._client = httpx.AsyncClient(base_url=self.base_url)
    return await self._client.post(path, json=payload, timeout=timeout)

  async def _post_ok(
      self, path: str, payload: dict[str, Any], timeout: float
  ) -> dict[str, Any]:
    response = await self._post(path, payload, timeout)
    response.raise_for_status()
    return response.json()

  async def start(self) -> None:
    """Boots a fresh container, or restores a handle into a new live one."""
    if self.env_id is None:
      result = await self._post_ok(
          "/start", {"task_name": self.task_name}, _START_TIMEOUT_SECONDS
      )
    else:
      result = await self._post_ok(
          "/fork", {"id": self.env_id}, _FORK_TIMEOUT_SECONDS
      )
    self.env_id = result["id"]

  async def fork(self) -> "TerminalBenchSandbox":
    """Snapshots this sandbox, which keeps running, and returns a handle."""
    result = await self._post_ok(
        "/fork", {"id": self.env_id}, _FORK_TIMEOUT_SECONDS
    )
    return TerminalBenchSandbox(
        self.task_name,
        env_id=result["id"],
        base_url=self.base_url,
        command_timeout=self.command_timeout,
    )

  async def execute(self, tool_call: tvclient_tool_call.ToolCall) -> str:
    """Runs a command or the task's tests and returns a canonical string."""
    try:
      if isinstance(tool_call, tvclient_tool_call.TestToolCall):
        return await self._run_tests()
      return await self._run_command(tool_call.command)
    except (httpx.HTTPError, KeyError, ValueError) as error:
      return f"{FAILURE_PREFIX}: {error!r}"

  async def _run_tests(self) -> str:
    response = await self._post(
        "/test", {"id": self.env_id}, _TEST_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
      return f"{FAILURE_PREFIX} tests: {response.text}"
    return PASSED if response.json()["passed"] else FAILED

  async def _run_command(self, command: str) -> str:
    before = await self._settled_pane()
    response = await self._post(
        "/execute",
        {
            "id": self.env_id,
            "command": command,
            "blocking": True,
            "timeout": self.command_timeout,
        },
        self.command_timeout + _COMMAND_TIMEOUT_MARGIN_SECONDS,
    )
    if response.status_code != 200:
      return f"{FAILURE_PREFIX} command: {response.text}"
    after = await self._settled_pane()
    return extract_new_output(before, after)

  async def _settled_pane(self) -> str:
    """Returns the pane once a prompt shows, or the latest after a wait."""
    deadline = time.monotonic() + _SETTLE_TIMEOUT_SECONDS
    while True:
      result = await self._post_ok(
          "/get_pane_logs", {"id": self.env_id}, _REQUEST_TIMEOUT_SECONDS
      )
      pane = result["logs"]
      lines = normalize_pane(pane)
      if lines and _BARE_PROMPT.match(lines[-1]):
        return pane
      if time.monotonic() >= deadline:
        return pane
      await asyncio.sleep(_SETTLE_POLL_SECONDS)

  async def stop(self) -> None:
    try:
      await self._post_ok(
          "/stop", {"id": self.env_id}, _REQUEST_TIMEOUT_SECONDS
      )
    finally:
      if self._client is not None:
        await self._client.aclose()
        self._client = None
