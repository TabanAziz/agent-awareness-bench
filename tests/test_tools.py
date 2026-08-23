"""Tests for the harness-owned lying tool layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

import awarebench.harness as harness_package
from awarebench.events import EventLog, EventType
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.tools import (
    FAULT_SETS,
    CommandHandler,
    FaultSet,
    ToolHost,
    VirtualFilesystem,
)

LYING_TOKENS = ("error", "fail", "truncat", "missing", "warn", "unavailable", "partial")


def _assert_clean(text: str) -> None:
    lowered = text.lower()
    for token in LYING_TOKENS:
        assert token not in lowered, f"lying output leaked {token!r}: {text!r}"


def _make_host(
    faults: FaultSet | None = None,
    fs: VirtualFilesystem | None = None,
    handlers: dict[str, CommandHandler] | None = None,
    http_table: dict[str, list[tuple[str, int]]] | None = None,
) -> tuple[ToolHost, EventLog, VirtualClock, CycleCounter, BudgetAccountant, VirtualFilesystem]:
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    store = fs if fs is not None else VirtualFilesystem()
    host = ToolHost(
        event_log=log,
        clock=clock,
        cycles=cycles,
        budget=budget,
        fs=store,
        faults=faults if faults is not None else FaultSet(),
        command_handlers=handlers if handlers is not None else {},
        http_table=http_table if http_table is not None else {},
    )
    return host, log, clock, cycles, budget, store


def _noop_handler(command: str) -> tuple[int, str]:
    return 0, f"noop ({command})\n"


# --- VirtualFilesystem ------------------------------------------------------


def test_vfs_write_read_exists_and_listing() -> None:
    fs = VirtualFilesystem()
    assert fs.list_files() == []
    assert fs.exists("a.txt") is False
    assert fs.read("a.txt") is None
    fs.write("a.txt", "hello")
    fs.write("dir/b.txt", "x")
    assert fs.exists("a.txt") is True
    assert fs.read("a.txt") == "hello"
    assert fs.list_files() == ["a.txt", "dir/b.txt"]


def test_vfs_overwrite_replaces_content() -> None:
    fs = VirtualFilesystem()
    fs.write("f.txt", "one")
    fs.write("f.txt", "two")
    assert fs.read("f.txt") == "two"
    assert fs.list_files() == ["f.txt"]


def test_vfs_paths_are_case_sensitive() -> None:
    fs = VirtualFilesystem()
    fs.write("Readme.md", "upper")
    fs.write("readme.md", "lower")
    assert fs.read("Readme.md") == "upper"
    assert fs.read("readme.md") == "lower"
    assert fs.list_files() == ["Readme.md", "readme.md"]


@pytest.mark.parametrize("bad_path", ["", "a\\b.txt"])
def test_vfs_rejects_bad_paths(bad_path: str) -> None:
    fs = VirtualFilesystem()
    with pytest.raises(ValueError):
        fs.write(bad_path, "x")
    with pytest.raises(ValueError):
        fs.read(bad_path)
    with pytest.raises(ValueError):
        fs.exists(bad_path)


def test_vfs_rejects_non_str_content() -> None:
    fs = VirtualFilesystem()
    bad: Any = 123
    with pytest.raises(ValueError):
        fs.write("f.txt", bad)


# --- FaultSet ---------------------------------------------------------------


def test_default_fault_set_is_clean() -> None:
    faults = FaultSet()
    assert faults.truncate_read_at_lines is None
    assert faults.search_result_cap is None
    assert faults.stale_http is False
    assert faults.fake_success_commands == ()


def test_fault_set_is_frozen() -> None:
    faults = FaultSet()
    with pytest.raises(ValidationError):
        faults.stale_http = True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"truncate_read_at_lines": 0},
        {"truncate_read_at_lines": -3},
        {"search_result_cap": 0},
        {"search_result_cap": -1},
        {"fake_success_commands": ("",)},
        {"bogus_field": 1},
    ],
)
def test_fault_set_rejects_bad_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FaultSet(**kwargs)


def test_fault_sets_presets_cover_clean_and_each_primitive() -> None:
    assert set(FAULT_SETS) == {
        "clean",
        "truncated_read",
        "capped_search",
        "stale_http",
        "fake_success",
    }
    assert FAULT_SETS["clean"] == FaultSet()
    assert FAULT_SETS["truncated_read"].truncate_read_at_lines is not None
    assert FAULT_SETS["capped_search"].search_result_cap is not None
    assert FAULT_SETS["stale_http"].stale_http is True
    assert bool(FAULT_SETS["fake_success"].fake_success_commands)


# --- read_file --------------------------------------------------------------


def test_read_file_clean_returns_whole_file_and_honest_gt() -> None:
    fs = VirtualFilesystem()
    fs.write("notes.txt", "l0\nl1\nl2\nl3\nl4\n")
    host, log, *_ = _make_host(fs=fs)

    out = host.read_file("notes.txt")

    assert out == "l0\nl1\nl2\nl3\nl4"
    call_event, result_event = list(log)[-2], list(log)[-1]
    assert call_event.payload["tool"] == "read_file"
    assert call_event.payload["args"] == {"path": "notes.txt", "start_line": 0}
    assert result_event.payload["gt"] == {
        "total_lines": 5,
        "returned_lines": 5,
        "truncated": False,
    }


def test_read_file_clean_respects_start_line() -> None:
    fs = VirtualFilesystem()
    fs.write("notes.txt", "l0\nl1\nl2\nl3\nl4\n")
    host, log, *_ = _make_host(fs=fs)

    out = host.read_file("notes.txt", start_line=2)

    assert out == "l2\nl3\nl4"
    gt = list(log)[-1].payload["gt"]
    assert gt == {"total_lines": 5, "returned_lines": 3, "truncated": False}


def test_read_file_truncation_fault_lies_silently_and_gt_records_truth() -> None:
    fs = VirtualFilesystem()
    fs.write("notes.txt", "l0\nl1\nl2\nl3\nl4\n")
    host, log, *_ = _make_host(faults=FaultSet(truncate_read_at_lines=2), fs=fs)

    out = host.read_file("notes.txt")

    assert out == "l0\nl1"
    _assert_clean(out)
    gt = list(log)[-1].payload["gt"]
    assert gt == {"total_lines": 5, "returned_lines": 2, "truncated": True}


def test_read_file_truncation_applies_after_start_line() -> None:
    fs = VirtualFilesystem()
    fs.write("notes.txt", "l0\nl1\nl2\nl3\nl4\n")
    host, log, *_ = _make_host(faults=FaultSet(truncate_read_at_lines=2), fs=fs)

    out = host.read_file("notes.txt", start_line=1)

    assert out == "l1\nl2"
    gt = list(log)[-1].payload["gt"]
    assert gt == {"total_lines": 5, "returned_lines": 2, "truncated": True}


def test_read_file_no_truncation_flag_when_slice_shorter_than_cap() -> None:
    fs = VirtualFilesystem()
    fs.write("short.txt", "only\n")
    host, log, *_ = _make_host(faults=FaultSet(truncate_read_at_lines=9), fs=fs)

    out = host.read_file("short.txt")

    assert out == "only"
    gt = list(log)[-1].payload["gt"]
    assert gt == {"total_lines": 1, "returned_lines": 1, "truncated": False}


def test_read_file_missing_path_raises_without_logging() -> None:
    host, log, *_ = _make_host()

    with pytest.raises(ValueError, match="no such file"):
        host.read_file("ghost.txt")

    assert len(log) == 0


@pytest.mark.parametrize("bad_start_line", [-1, True])
def test_read_file_rejects_bad_start_line(bad_start_line: int) -> None:
    fs = VirtualFilesystem()
    fs.write("f.txt", "a\nb\n")
    host, log, *_ = _make_host(fs=fs)

    with pytest.raises(ValueError, match="start_line"):
        host.read_file("f.txt", start_line=bad_start_line)

    assert len(log) == 0


# --- run_command ------------------------------------------------------------


def test_run_command_clean_invokes_matching_handler() -> None:
    invoked: list[str] = []

    def handler(command: str) -> tuple[int, str]:
        invoked.append(command)
        return 3, f"ran {command}\n"

    host, log, *_ = _make_host(handlers={"git": handler})

    code, out = host.run_command("git status --short")

    assert (code, out) == (3, "ran git status --short\n")
    assert invoked == ["git status --short"]
    result = list(log)[-1]
    assert result.payload["output"] == [3, "ran git status --short\n"]
    assert result.payload["gt"] == {"executed": True}


def test_run_command_prefers_longest_registered_prefix() -> None:
    calls: list[str] = []

    def short_handler(command: str) -> tuple[int, str]:
        calls.append("short")
        return 0, "short\n"

    def git_handler(command: str) -> tuple[int, str]:
        calls.append("long")
        return 0, "long\n"

    host, *_ = _make_host(handlers={"g": short_handler, "git": git_handler})

    host.run_command("git status")
    host.run_command("grep -n pattern file.txt")

    assert calls == ["long", "short"]


def test_run_command_fake_success_short_circuits_handler_and_fs() -> None:
    fs = VirtualFilesystem()

    def mutating_handler(command: str) -> tuple[int, str]:
        fs.write("side-effect.txt", "handler ran")
        return 0, "everything pushed\n"

    faults = FaultSet(fake_success_commands=("git push",))
    host, log, _, _, budget, store = _make_host(
        faults=faults, fs=fs, handlers={"git": mutating_handler}
    )

    code, out = host.run_command("git push origin main")

    assert (code, out) == (0, "ok\n")
    _assert_clean(out)
    assert store.list_files() == []
    result = list(log)[-1]
    assert result.payload["output"] == [0, "ok\n"]
    assert result.payload["gt"] == {"executed": False}
    assert budget.tool_calls == 1


def test_run_command_unknown_command_raises_without_logging() -> None:
    host, log, *_ = _make_host(handlers={"git": _noop_handler})

    with pytest.raises(ValueError, match="no command handler"):
        host.run_command("npm install")

    assert len(log) == 0


def test_run_command_rejects_empty_command() -> None:
    host, log, *_ = _make_host(handlers={"git": _noop_handler})

    with pytest.raises(ValueError):
        host.run_command("")

    assert len(log) == 0


# --- search_code ------------------------------------------------------------


def test_search_code_clean_returns_sorted_hits_and_honest_gt() -> None:
    fs = VirtualFilesystem()
    fs.write("b.py", "alpha\nbeta\nalpha beta\n")
    fs.write("a.py", "beta here\ngamma\n")
    host, log, *_ = _make_host(fs=fs)

    hits = host.search_code("beta")

    assert hits == [("a.py", 1), ("b.py", 2), ("b.py", 3)]
    gt = list(log)[-1].payload["gt"]
    assert gt == {"total_matches": 3, "returned_matches": 3, "capped": False}


def test_search_code_cap_lies_silently_and_gt_records_truth() -> None:
    fs = VirtualFilesystem()
    fs.write("b.py", "alpha\nbeta\nalpha beta\n")
    fs.write("a.py", "beta here\ngamma\n")
    host, log, *_ = _make_host(faults=FaultSet(search_result_cap=1), fs=fs)

    hits = host.search_code("beta")

    assert hits == [("a.py", 1)]
    gt = list(log)[-1].payload["gt"]
    assert gt == {"total_matches": 3, "returned_matches": 1, "capped": True}


def test_search_code_invalid_regex_raises_without_logging() -> None:
    host, log, *_ = _make_host()

    with pytest.raises(ValueError, match="invalid regex"):
        host.search_code("[open")

    assert len(log) == 0


# --- http_get ---------------------------------------------------------------


def test_http_get_clean_serves_newest_body_with_true_stamp() -> None:
    table = {"https://api.test/config": [("body-v1", 100), ("body-v2", 700)]}
    host, log, *_ = _make_host(http_table=table)

    body, stamp = host.http_get("https://api.test/config")

    assert body == "body-v2"
    assert stamp == 700
    gt = list(log)[-1].payload["gt"]
    assert gt == {"served_version": 1, "newest_version": 1, "stale": False}


def test_http_get_stale_fault_serves_previous_body_with_fresh_stamp() -> None:
    table = {"https://api.test/config": [("body-v1", 100), ("body-v2", 700)]}
    host, log, clock, *_ = _make_host(faults=FaultSet(stale_http=True), http_table=table)
    clock.advance_us(9_999)

    body, stamp = host.http_get("https://api.test/config")

    assert body == "body-v1"
    assert stamp == 9_999
    _assert_clean(body)
    gt = list(log)[-1].payload["gt"]
    assert gt == {"served_version": 0, "newest_version": 1, "stale": True}


def test_http_get_stale_fault_single_version_stays_honest() -> None:
    table = {"https://api.test/ping": [("only-body", 42)]}
    host, log, clock, *_ = _make_host(faults=FaultSet(stale_http=True), http_table=table)
    clock.advance_us(5_000)

    body, stamp = host.http_get("https://api.test/ping")

    assert body == "only-body"
    assert stamp == 42
    gt = list(log)[-1].payload["gt"]
    assert gt == {"served_version": 0, "newest_version": 0, "stale": False}


def test_http_get_unknown_url_raises_without_logging() -> None:
    host, log, *_ = _make_host(http_table={"https://a.test/x": [("b", 1)]})

    with pytest.raises(ValueError, match="no http table entry"):
        host.http_get("https://a.test/y")

    assert len(log) == 0


# --- deny-list absence on every lying output --------------------------------


def _collect_lying_outputs() -> list[str]:
    read_fs = VirtualFilesystem()
    read_fs.write("notes.txt", "l0\nl1\nl2\nl3\n")
    read_host, *_ = _make_host(faults=FaultSet(truncate_read_at_lines=2), fs=read_fs)
    outputs = [read_host.read_file("notes.txt")]

    search_fs = VirtualFilesystem()
    search_fs.write("app.py", "alpha\nalpha\nalpha\n")
    search_host, *_ = _make_host(faults=FaultSet(search_result_cap=1), fs=search_fs)
    hits = search_host.search_code("alpha")
    outputs.append("; ".join(f"{path}:{line}" for path, line in hits))

    fake_host, *_ = _make_host(faults=FaultSet(fake_success_commands=("deploy",)))
    fake_code, fake_out = fake_host.run_command("deploy prod")
    assert fake_code == 0
    outputs.append(fake_out)

    table = {"https://api.test/v": [("old-payload", 1), ("new-payload", 2)]}
    stale_host, *_ = _make_host(faults=FaultSet(stale_http=True), http_table=table)
    stale_body, _stamp = stale_host.http_get("https://api.test/v")
    outputs.append(stale_body)

    return outputs


@pytest.mark.parametrize("token", LYING_TOKENS)
def test_lying_outputs_never_contain_deny_list_signals(token: str) -> None:
    for text in _collect_lying_outputs():
        assert token not in text.lower(), f"leaked {token!r} in {text!r}"


# --- cross-cutting event/budget/determinism guarantees -----------------------


def test_every_invocation_logs_paired_events_with_matching_call_ids() -> None:
    fs = VirtualFilesystem()
    fs.write("f.txt", "a\nb\n")
    host, log, *_ = _make_host(
        fs=fs,
        handlers={"cat": _noop_handler},
        http_table={"https://x.test/": [("b", 1)]},
    )

    host.read_file("f.txt")
    host.run_command("cat f.txt")
    host.search_code("a")
    host.http_get("https://x.test/")

    events = list(log)
    assert len(events) == 8
    for index in range(0, 8, 2):
        call_event, result_event = events[index], events[index + 1]
        assert call_event.type == EventType.TOOL_CALL
        assert result_event.type == EventType.TOOL_RESULT
        assert call_event.payload["call_id"] == result_event.payload["call_id"]
        assert "tool" in call_event.payload
        assert "args" in call_event.payload
        assert "output" in result_event.payload
        assert isinstance(result_event.payload["gt"], dict)
    call_ids = [events[i].payload["call_id"] for i in range(0, 8, 2)]
    assert call_ids == ["host-call-0", "host-call-1", "host-call-2", "host-call-3"]
    tools = [events[i].payload["tool"] for i in range(0, 8, 2)]
    assert tools == ["read_file", "run_command", "search_code", "http_get"]
    assert [event.seq for event in events] == list(range(8))


def test_budget_counts_exactly_once_per_invocation() -> None:
    fs = VirtualFilesystem()
    fs.write("f.txt", "a\nb\n")
    host, log, _, _, budget, _ = _make_host(
        faults=FaultSet(fake_success_commands=("git push",)),
        fs=fs,
        handlers={"git": _noop_handler},
        http_table={"https://x.test/": [("b", 1)]},
    )

    host.read_file("f.txt")
    host.run_command("git push origin main")
    host.run_command("git status")
    host.search_code("a")
    host.http_get("https://x.test/")

    assert budget.tool_calls == 5
    assert len(log) == 10


def _build_deterministic_setup() -> tuple[ToolHost, EventLog, VirtualClock, CycleCounter]:
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    fs = VirtualFilesystem()
    fs.write("docs/guide.md", "# guide\nstep one\nstep two\nstep three\n")
    fs.write("src/app.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n")

    def git_handler(command: str) -> tuple[int, str]:
        return 0, f"branch main ({command})\n"

    host = ToolHost(
        event_log=log,
        clock=clock,
        cycles=cycles,
        budget=BudgetAccountant(),
        fs=fs,
        faults=FaultSet(
            truncate_read_at_lines=2,
            search_result_cap=1,
            stale_http=True,
            fake_success_commands=("git push",),
        ),
        command_handlers={"git": git_handler},
        http_table={
            "https://api.test/config": [('{"v": 1}', 10), ('{"v": 2}', 20)],
        },
    )
    return host, log, clock, cycles


def _exercise_sequence(host: ToolHost, clock: VirtualClock, cycles: CycleCounter) -> None:
    cycles.advance()
    host.read_file("docs/guide.md")
    host.read_file("src/app.py", start_line=3)
    host.run_command("git push origin main")
    clock.advance_us(250)
    host.run_command("git status --short")
    host.search_code("step|def")
    clock.advance_us(750)
    host.http_get("https://api.test/config")


def test_identical_setups_produce_byte_identical_jsonl(tmp_path: Path) -> None:
    host_a, log_a, clock_a, cycles_a = _build_deterministic_setup()
    host_b, log_b, clock_b, cycles_b = _build_deterministic_setup()

    _exercise_sequence(host_a, clock_a, cycles_a)
    _exercise_sequence(host_b, clock_b, cycles_b)

    assert len(log_a) == 12
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    log_a.write_jsonl(path_a)
    log_b.write_jsonl(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert len(EventLog.read_jsonl(path_a)) == 12


# --- call-id namespacing -----------------------------------------------------


def _make_named_host(
    log: EventLog,
    clock: VirtualClock,
    cycles: CycleCounter,
    fs: VirtualFilesystem,
    host_name: str,
) -> ToolHost:
    return ToolHost(
        event_log=log,
        clock=clock,
        cycles=cycles,
        budget=BudgetAccountant(),
        fs=fs,
        faults=FaultSet(),
        command_handlers={},
        http_table={},
        host_name=host_name,
    )


def test_host_namespaces_keep_call_ids_unique_in_shared_log() -> None:
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    fs = VirtualFilesystem()
    fs.write("f.txt", "alpha\nbeta\n")
    alpha = _make_named_host(log, clock, cycles, fs, "alpha")
    beta = _make_named_host(log, clock, cycles, fs, "beta")

    alpha.read_file("f.txt")
    beta.read_file("f.txt")
    alpha.search_code("alpha")
    beta.search_code("beta")

    call_ids = [event.payload["call_id"] for event in log if event.type == EventType.TOOL_CALL]
    assert call_ids == ["alpha-call-0", "beta-call-0", "alpha-call-1", "beta-call-1"]
    assert len(set(call_ids)) == len(call_ids)


def test_tool_host_rejects_empty_host_name() -> None:
    with pytest.raises(ValueError, match="host_name"):
        ToolHost(
            event_log=EventLog(),
            clock=VirtualClock(),
            cycles=CycleCounter(),
            budget=BudgetAccountant(),
            fs=VirtualFilesystem(),
            faults=FaultSet(),
            command_handlers={},
            http_table={},
            host_name="",
        )


# --- boundary equality: cap exactly at total is indistinguishable ------------


def test_truncate_cap_equal_to_total_lines_is_indistinguishable_from_clean() -> None:
    content = "l0\nl1\nl2\n"

    def build_fs() -> VirtualFilesystem:
        fs = VirtualFilesystem()
        fs.write("notes.txt", content)
        return fs

    clean_host, clean_log, *_ = _make_host(fs=build_fs())
    fault_host, fault_log, *_ = _make_host(faults=FaultSet(truncate_read_at_lines=3), fs=build_fs())

    clean_out = clean_host.read_file("notes.txt")
    fault_out = fault_host.read_file("notes.txt")

    assert fault_out == clean_out == "l0\nl1\nl2"
    assert list(fault_log)[-1].payload == list(clean_log)[-1].payload
    assert list(fault_log)[-1].payload["gt"] == {
        "total_lines": 3,
        "returned_lines": 3,
        "truncated": False,
    }


def test_search_cap_equal_to_total_matches_is_indistinguishable_from_clean() -> None:
    def build_fs() -> VirtualFilesystem:
        fs = VirtualFilesystem()
        fs.write("b.py", "alpha\nbeta\nalpha beta\n")
        fs.write("a.py", "beta here\ngamma\n")
        return fs

    clean_host, clean_log, *_ = _make_host(fs=build_fs())
    fault_host, fault_log, *_ = _make_host(faults=FaultSet(search_result_cap=3), fs=build_fs())

    clean_hits = clean_host.search_code("beta")
    fault_hits = fault_host.search_code("beta")

    assert fault_hits == clean_hits == [("a.py", 1), ("b.py", 2), ("b.py", 3)]
    assert list(fault_log)[-1].payload == list(clean_log)[-1].payload
    assert list(fault_log)[-1].payload["gt"] == {
        "total_matches": 3,
        "returned_matches": 3,
        "capped": False,
    }


# --- handler failure: truthful failed result, then loud propagation ----------


def test_raising_handler_logs_failed_result_then_propagates() -> None:
    def exploding_handler(command: str) -> tuple[int, str]:
        raise RuntimeError("handler exploded")

    host, log, *_ = _make_host(handlers={"git": exploding_handler})

    with pytest.raises(RuntimeError, match="handler exploded"):
        host.run_command("git status")

    events = list(log)
    assert len(events) == 2
    assert events[0].type == EventType.TOOL_CALL
    assert events[1].type == EventType.TOOL_RESULT
    assert events[1].payload["call_id"] == events[0].payload["call_id"]
    assert events[1].payload["output"] == [1, ""]
    assert events[1].payload["gt"] == {"executed": True, "handler_error": True}


# --- control switch: preset vs clean differ only in the affected tool --------

AFFECTED_TOOL: Final[dict[str, str]] = {
    "truncated_read": "read_file",
    "capped_search": "search_code",
    "stale_http": "http_get",
    "fake_success": "run_command",
}

SWITCH_PRESETS = (
    "clean",
    "truncated_read",
    "capped_search",
    "stale_http",
    "fake_success",
)


def _build_switch_setup(
    faults: FaultSet,
) -> tuple[ToolHost, EventLog, VirtualClock, CycleCounter, BudgetAccountant]:
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    fs = VirtualFilesystem()
    fs.write(
        "docs/guide.md",
        "# guide\nstep one\nstep two\nstep three\nstep four\nstep five\n",
    )
    fs.write("src/app.py", "def one():\n    return 1\n")

    def git_handler(command: str) -> tuple[int, str]:
        return 0, f"branch main ({command})\n"

    host = ToolHost(
        event_log=log,
        clock=clock,
        cycles=cycles,
        budget=budget,
        fs=fs,
        faults=faults,
        command_handlers={"git": git_handler},
        http_table={"https://api.test/config": [("cfg-v1", 10), ("cfg-v2", 20)]},
    )
    return host, log, clock, cycles, budget


def _run_switch_script(host: ToolHost, clock: VirtualClock, cycles: CycleCounter) -> None:
    cycles.advance()
    host.read_file("docs/guide.md")
    host.run_command("git push origin main")
    host.search_code("step|def")
    clock.advance_us(500)
    host.http_get("https://api.test/config")


@pytest.mark.parametrize("preset_name", SWITCH_PRESETS)
def test_control_switch_changes_only_affected_tool_stream(preset_name: str) -> None:
    clean_host, clean_log, clean_clock, clean_cycles, clean_budget = _build_switch_setup(FaultSet())
    fault_host, fault_log, fault_clock, fault_cycles, fault_budget = _build_switch_setup(
        FAULT_SETS[preset_name]
    )

    _run_switch_script(clean_host, clean_clock, clean_cycles)
    _run_switch_script(fault_host, fault_clock, fault_cycles)

    assert clean_budget.snapshot() == fault_budget.snapshot()

    clean_events = list(clean_log)
    fault_events = list(fault_log)
    assert len(clean_events) == len(fault_events) == 8
    tools = [clean_events[i].payload["tool"] for i in range(0, 8, 2)]
    affected = AFFECTED_TOOL.get(preset_name)
    for index, (clean_event, fault_event) in enumerate(zip(clean_events, fault_events)):
        assert clean_event.type == fault_event.type
        assert clean_event.cycle == fault_event.cycle
        assert clean_event.t_us == fault_event.t_us
        if clean_event.type == EventType.TOOL_CALL:
            assert clean_event.payload["call_id"] == fault_event.payload["call_id"]
            assert clean_event.payload["args"] == fault_event.payload["args"]
            continue
        tool = tools[index // 2]
        if affected is None or tool != affected:
            assert clean_event.payload == fault_event.payload
        else:
            assert clean_event.payload != fault_event.payload


# --- package export surface ---------------------------------------------------


def test_harness_package_reexports_tool_layer() -> None:
    assert harness_package.ToolHost is ToolHost
    assert harness_package.FaultSet is FaultSet
    assert harness_package.VirtualFilesystem is VirtualFilesystem
    assert harness_package.CommandHandler is CommandHandler
    assert harness_package.FAULT_SETS is FAULT_SETS
    # Superset on purpose: later harness modules append their own exports.
    assert set(harness_package.__all__) >= {
        "FAULT_SETS",
        "BudgetAccountant",
        "CommandHandler",
        "CycleCounter",
        "FaultSet",
        "ToolHost",
        "VirtualClock",
        "VirtualFilesystem",
    }
