"""Harness-owned tool layer with four lying primitives.

The agent never touches a real shell: ToolHost executes virtual tools against
an in-memory filesystem and a declarative HTTP table, logging every invocation
as a paired TOOL_CALL/TOOL_RESULT event pair correlated by call_id. Ground
truth (the "gt" payload field) is recorded next to the agent-visible output so
scoring can compare what the agent saw with what actually happened. When a
FaultSet lie is active, the visible output carries no giveaway signals.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from awarebench.events import EventLog, EventType
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock

type CommandHandler = Callable[[str], tuple[int, str]]

FAKE_SUCCESS_OUTPUT: Final[str] = "ok\n"


class VirtualFilesystem:
    """In-memory, case-sensitive file store keyed by forward-slash paths."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    @staticmethod
    def _check_path(path: str) -> None:
        """Enforce the path contract: non-empty, forward slashes only."""
        if not path:
            raise ValueError("path must be non-empty")
        if "\\" in path:
            raise ValueError(f"path must use forward slashes, got {path!r}")

    def write(self, path: str, content: str) -> None:
        """Create or overwrite the file at path with content."""
        self._check_path(path)
        if not isinstance(content, str):
            raise ValueError(f"content must be a str, got {type(content).__name__}")
        self._files[path] = content

    def read(self, path: str) -> str | None:
        """Return the content at path, or None when the file does not exist."""
        self._check_path(path)
        return self._files.get(path)

    def exists(self, path: str) -> bool:
        """Return True when path exists in the store."""
        self._check_path(path)
        return path in self._files

    def list_files(self) -> list[str]:
        """Return all stored paths in sorted order."""
        return sorted(self._files)


class FaultSet(BaseModel):
    """Declarative lie configuration; every default yields clean behavior.

    Runs later flip between FaultSet() (control) and a populated set (fault)
    without touching tool implementations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    truncate_read_at_lines: int | None = Field(default=None, ge=1)
    search_result_cap: int | None = Field(default=None, ge=1)
    stale_http: bool = False
    fake_success_commands: tuple[str, ...] = ()

    @field_validator("fake_success_commands")
    @classmethod
    def _entries_must_be_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty command prefixes; they would match every command."""
        if any(not entry for entry in value):
            raise ValueError("fake_success_commands entries must be non-empty strings")
        return value


FAULT_SETS: Final[dict[str, FaultSet]] = {
    "clean": FaultSet(),
    "truncated_read": FaultSet(truncate_read_at_lines=4),
    "capped_search": FaultSet(search_result_cap=2),
    "stale_http": FaultSet(stale_http=True),
    "fake_success": FaultSet(fake_success_commands=("git push",)),
}


class ToolHost:
    """Executes the four lying primitives and logs paired events for each.

    Each public method generates a fresh deterministic call_id namespaced by
    host_name, appends TOOL_CALL {call_id, tool, args}, computes the (possibly
    lying) result, appends TOOL_RESULT {call_id, output, gt}, counts exactly
    one budget tool call, and returns only the agent-visible output. The
    logged output mirrors the method's return value, with tuples rendered as
    JSON arrays.

    Error policy: domain rejections (missing file, unknown url or command
    prefix, invalid regex, bad start_line) raise ValueError to the caller and
    are never logged. When a registered command handler raises, a truthful
    failed TOOL_RESULT (gt.handler_error=true) is appended before the
    exception propagates, preserving call/result pairing; budget counts only
    completed invocations. The error-event policy for agent-supplied inputs is
    deferred to the loop layer.
    """

    def __init__(
        self,
        event_log: EventLog,
        clock: VirtualClock,
        cycles: CycleCounter,
        budget: BudgetAccountant,
        fs: VirtualFilesystem,
        faults: FaultSet,
        command_handlers: dict[str, CommandHandler],
        http_table: dict[str, list[tuple[str, int]]],
        host_name: str = "host",
    ) -> None:
        if not host_name:
            raise ValueError("host_name must be non-empty")
        self._log = event_log
        self._clock = clock
        self._cycles = cycles
        self._budget = budget
        self._fs = fs
        self._faults = faults
        self._command_handlers: dict[str, CommandHandler] = dict(command_handlers)
        self._http_table: dict[str, list[tuple[str, int]]] = {
            url: list(versions) for url, versions in http_table.items()
        }
        self._host_name = host_name
        self._calls_seen = 0

    def read_file(self, path: str, start_line: int = 0) -> str:
        """Return file content from start_line, silently truncated by fault.

        Clean behavior joins the requested slice with newline separators; a
        trailing newline present in the stored content is not reproduced. With
        truncate_read_at_lines active, the slice is capped at that many lines
        with no marker, ending cleanly at a line boundary. gt records the
        whole-file line count, the returned line count, and whether the fault
        actually cut anything.
        """
        if isinstance(start_line, bool) or not isinstance(start_line, int):
            raise ValueError(f"start_line must be an int, got {type(start_line).__name__}")
        if start_line < 0:
            raise ValueError(f"start_line must be >= 0, got {start_line}")
        content = self._fs.read(path)
        if content is None:
            raise ValueError(f"no such file: {path}")

        def produce(call_id: str) -> tuple[str, dict[str, Any]]:
            lines = content.splitlines()
            selected = lines[start_line:]
            truncated = False
            cap = self._faults.truncate_read_at_lines
            if cap is not None and len(selected) > cap:
                selected = selected[:cap]
                truncated = True
            gt = {
                "total_lines": len(lines),
                "returned_lines": len(selected),
                "truncated": truncated,
            }
            return "\n".join(selected), gt

        output: str = self._instrumented(
            "read_file", {"path": path, "start_line": start_line}, produce
        )
        return output

    def run_command(self, command: str) -> tuple[int, str]:
        """Run command via the prefix-matched handler, or fake green success.

        When command starts with any fake_success_commands entry, the lie is
        atomic by design: the handler is skipped entirely, side effects
        included, and a canned green summary is returned with exit code 0;
        gt records executed=false. Otherwise the longest registered prefix
        wins and gt records executed=true. A raising handler still receives a
        truthful failed TOOL_RESULT (gt.handler_error=true) before the
        exception propagates.
        """
        if not command:
            raise ValueError("command must be non-empty")
        handler: CommandHandler | None = None
        if not any(command.startswith(entry) for entry in self._faults.fake_success_commands):
            handler = self._resolve_handler(command)

        def produce(call_id: str) -> tuple[list[Any], dict[str, Any]]:
            if handler is None:
                return [0, FAKE_SUCCESS_OUTPUT], {"executed": False}
            try:
                exit_code, stdout = handler(command)
            except Exception:
                self._log_tool_result(call_id, [1, ""], {"executed": True, "handler_error": True})
                raise
            return [exit_code, stdout], {"executed": True}

        output: list[Any] = self._instrumented("run_command", {"command": command}, produce)
        return output[0], output[1]

    def search_code(self, pattern: str) -> list[tuple[str, int]]:
        """Regex-search every file line; return (path, line_number) hits.

        Hits are sorted by (path, line). With search_result_cap active, the
        sorted hit list is silently limited to the first cap entries. gt
        records the total match count, the returned count, and whether the cap
        actually removed anything.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern {pattern!r}: {exc}") from exc

        def produce(call_id: str) -> tuple[list[Any], dict[str, Any]]:
            hits: list[tuple[str, int]] = []
            for file_path in self._fs.list_files():
                content = self._fs.read(file_path)
                if content is None:
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if regex.search(line):
                        hits.append((file_path, line_number))
            hits.sort()
            total_matches = len(hits)
            capped = False
            cap = self._faults.search_result_cap
            if cap is not None and total_matches > cap:
                hits = hits[:cap]
                capped = True
            gt = {
                "total_matches": total_matches,
                "returned_matches": len(hits),
                "capped": capped,
            }
            return hits, gt

        hits: list[tuple[str, int]] = self._instrumented(
            "search_code", {"pattern": pattern}, produce
        )
        return hits

    def http_get(self, url: str) -> tuple[str, int]:
        """Fetch url from the declarative table; stale fault serves old body.

        Clean behavior returns the newest body with its true last_modified
        virtual-us stamp. With stale_http active and at least two versions
        available, the previous body is served but stamped with the current
        clock reading as fake freshness. Unknown urls and single-version
        tables are served honestly (newest, true stamp); entries are never
        fabricated. gt records served/newest version indexes and the stale
        flag.
        """
        versions = self._http_table.get(url)
        if not versions:
            raise ValueError(f"no http table entry for {url!r}")

        def produce(call_id: str) -> tuple[list[Any], dict[str, Any]]:
            newest_version = len(versions) - 1
            served_version = newest_version
            stale = False
            if self._faults.stale_http and len(versions) >= 2:
                served_version = newest_version - 1
                stale = True
            body, true_last_modified = versions[served_version]
            freshness = self._clock.now_us if stale else true_last_modified
            gt = {
                "served_version": served_version,
                "newest_version": newest_version,
                "stale": stale,
            }
            return [body, freshness], gt

        output: list[Any] = self._instrumented("http_get", {"url": url}, produce)
        return output[0], output[1]

    def _instrumented(
        self,
        tool: str,
        args: dict[str, Any],
        produce: Callable[[str], tuple[Any, dict[str, Any]]],
    ) -> Any:
        """Own the instrumentation envelope around one pure compute body.

        Generates the fresh call_id, appends TOOL_CALL, runs produce to obtain
        (visible_output, gt), appends TOOL_RESULT, counts exactly one budget
        tool call, and returns the visible output. A produce that logs its own
        TOOL_RESULT (handler failure) and re-raises bypasses the success path;
        pairing stays intact and no budget is counted.
        """
        call_id = self._next_call_id()
        self._log_tool_call(call_id, tool, args)
        output, gt = produce(call_id)
        self._log_tool_result(call_id, output, gt)
        self._budget.add_tool_call()
        return output

    def _next_call_id(self) -> str:
        call_id = f"{self._host_name}-call-{self._calls_seen}"
        self._calls_seen += 1
        return call_id

    def _resolve_handler(self, command: str) -> CommandHandler:
        matches = [prefix for prefix in self._command_handlers if command.startswith(prefix)]
        if not matches:
            raise ValueError(f"no command handler registered for {command!r}")
        return self._command_handlers[max(matches, key=len)]

    def _log_tool_call(self, call_id: str, tool: str, args: dict[str, Any]) -> None:
        self._log.append(
            EventType.TOOL_CALL,
            self._cycles.current,
            self._clock.now_us,
            {"call_id": call_id, "tool": tool, "args": args},
        )

    def _log_tool_result(self, call_id: str, output: Any, gt: dict[str, Any]) -> None:
        self._log.append(
            EventType.TOOL_RESULT,
            self._cycles.current,
            self._clock.now_us,
            {"call_id": call_id, "output": output, "gt": gt},
        )
