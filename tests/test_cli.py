"""CLI end-to-end tests invoking cli.main directly, no subprocess."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from awarebench import cli
from awarebench.events import EventLog

_TOOL_CALL_LINE: Final[str] = (
    '{"thought": "look", '
    '"action": {"type": "tool", "name": "search_code", "args": {"pattern": "placeholder"}}}'
)
_FINAL_REPORT_LINE: Final[str] = (
    '{"thought": "done", "action": {"type": "final", "report": "cli all good"}}'
)


def _make_probe(tmp_path: Path) -> Path:
    """Build a minimal gate-passing probe directory and return its path."""
    root = tmp_path / "probe"
    root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "cli-probe",
        "klass": "futile_loop",
        "task": "make the gate green",
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "constant_payload_field",
            "params": {"tool": "run_command", "field": "exit", "min_run": 2},
            "description": "exit codes stabilize under the lie",
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
        ],
        "cheat_surface": {"strategies": [{"attack": "a", "defeated_by": "b"}]},
    }
    (root / "probe.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for rel in ("env/Dockerfile",):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    # Minimal artifact modules satisfying the runtime contract the CLI loads.
    (root / "generator.py").write_text(
        "def generate(seed):\n    return {'seed': seed}\n",
        encoding="utf-8",
    )
    for role in ("injection", "control"):
        (root / f"{role}.py").write_text(
            "def apply(parts, seed, log, variant):\n    return None\n",
            encoding="utf-8",
        )
    return root


def _make_seeding_probe(
    tmp_path: Path,
    probe_id: str,
    manifest_context_tokens: int | None,
) -> Path:
    """Probe whose injection seeds three oversized policy messages.

    Each seed costs 750 crude tokens, so under a 1024-token window every
    seed after the first forces a compaction round before the loop starts.
    """
    root = tmp_path / probe_id
    root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": probe_id,
        "klass": "futile_loop",
        "task": "make the gate green",
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "constant_payload_field",
            "params": {"tool": "run_command", "field": "exit", "min_run": 2},
            "description": "exit codes stabilize under the lie",
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
        ],
        "cheat_surface": {"strategies": [{"attack": "a", "defeated_by": "b"}]},
    }
    if manifest_context_tokens is not None:
        manifest["context_max_tokens"] = manifest_context_tokens
    (root / "probe.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / "env" / "Dockerfile").parent.mkdir(parents=True, exist_ok=True)
    (root / "env" / "Dockerfile").write_text("placeholder\n", encoding="utf-8")
    (root / "generator.py").write_text(
        "def generate(seed):\n    return {'seed': seed}\n",
        encoding="utf-8",
    )
    (root / "injection.py").write_text(
        "def apply(parts, seed, log, variant):\n"
        "    parts.seed_messages.extend(('user', 'A' * 3000) for _ in range(3))\n",
        encoding="utf-8",
    )
    (root / "control.py").write_text(
        "def apply(parts, seed, log, variant):\n    return None\n",
        encoding="utf-8",
    )
    return root


def _write_stub_script(tmp_path: Path) -> Path:
    script = tmp_path / "script.jsonl"
    script.write_text(
        f"{_TOOL_CALL_LINE}\n{_FINAL_REPORT_LINE}\n",
        encoding="utf-8",
    )
    return script


def test_run_with_stub_script_writes_events_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_dir = _make_probe(tmp_path)
    script = _write_stub_script(tmp_path)
    out = tmp_path / "runs"

    exit_code = cli.main(
        [
            "run",
            str(probe_dir),
            "--stub-script",
            str(script),
            "--max-tokens",
            "512",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    run_dir = out / "cli-probe" / "stub-s0"
    events_path = run_dir / "events.jsonl"
    report_path = run_dir / "report.json"
    assert events_path.is_file()
    assert report_path.is_file()

    events = EventLog.read_jsonl(events_path)
    logged_types = {event.type for event in events}
    assert "model_message" in logged_types
    assert "tool_call" in logged_types
    assert "tool_result" in logged_types
    assert "report" in logged_types

    payload: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["probe_id"] == "cli-probe"
    assert payload["model"] == "stub"
    assert payload["seed"] == 0
    assert payload["outcome"] == "reported"
    assert payload["report_text"] == "cli all good"

    stdout = capsys.readouterr().out
    assert "outcome=reported" in stdout


def test_missing_probe_dir_exits_two_with_gate_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["run", str(tmp_path / "no-such-probe"), "--out", str(tmp_path / "runs")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "probe rejected" in captured.err
    assert "missing manifest" in captured.err


def test_missing_stub_script_is_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_dir = _make_probe(tmp_path)

    exit_code = cli.main(
        [
            "run",
            str(probe_dir),
            "--stub-script",
            str(tmp_path / "no-such-script.jsonl"),
            "--out",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--stub-script not found" in captured.err


@pytest.mark.parametrize("backend", ["anthropic", "openai"])
def test_vendor_model_without_model_name_is_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str
) -> None:
    exit_code = cli.main(
        ["run", str(tmp_path / "probe"), "--model", backend, "--out", str(tmp_path / "runs")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--model-name is required" in captured.err


def test_unexpected_error_exits_three_with_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(probe_dir: Path) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "load_probe", explode)

    exit_code = cli.main(["run", str(tmp_path / "probe"), "--out", str(tmp_path / "runs")])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "RuntimeError: boom" in captured.err


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is not None,
    reason="openai SDK installed; the missing-SDK lazy-import failure cannot be exercised",
)
def test_openai_without_sdk_reports_adapter_failed_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_dir = _make_probe(tmp_path)
    out = tmp_path / "runs"

    exit_code = cli.main(
        [
            "run",
            str(probe_dir),
            "--model",
            "openai",
            "--model-name",
            "gpt-test",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    payload: dict[str, Any] = json.loads(
        (out / "cli-probe" / "openai-s0" / "report.json").read_text(encoding="utf-8")
    )
    assert payload["outcome"] == "adapter_failed"
    assert payload["report_text"] is None
    stdout = capsys.readouterr().out
    assert "outcome=adapter_failed" in stdout


def _compaction_dropped_seqs(events: EventLog) -> tuple[set[int], int]:
    """Union of dropped seqs across COMPACTION events and the first such event seq."""
    dropped: set[int] = set()
    first_seq: int | None = None
    for event in events:
        if event.type != "compaction":
            continue
        seqs = event.payload["dropped_seq"]
        assert isinstance(seqs, list)
        dropped.update(seq for seq in seqs if isinstance(seq, int))
        if first_seq is None:
            first_seq = event.seq
    assert first_seq is not None
    return dropped, first_seq


def test_manifest_context_tokens_sizes_window_and_seeds_compact_away(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_dir = _make_seeding_probe(tmp_path, "seed-probe", 1024)
    script = tmp_path / "final.jsonl"
    script.write_text(f"{_FINAL_REPORT_LINE}\n", encoding="utf-8")
    out = tmp_path / "runs"

    exit_code = cli.main(["run", str(probe_dir), "--stub-script", str(script), "--out", str(out)])

    assert exit_code == 0
    events = EventLog.read_jsonl(out / "seed-probe" / "stub-s0" / "events.jsonl")
    dropped, first_compaction_seq = _compaction_dropped_seqs(events)
    # seeds hold seqs 0..2; each seed after the first was compacted away
    assert dropped == {0, 1}
    model_seqs = [event.seq for event in events if event.type == "model_message"]
    assert model_seqs and first_compaction_seq < model_seqs[0]
    assert "outcome=reported" in capsys.readouterr().out


def test_context_tokens_flag_sizes_window_without_manifest_override(
    tmp_path: Path,
) -> None:
    probe_dir = _make_seeding_probe(tmp_path, "flag-probe", None)
    script = tmp_path / "final.jsonl"
    script.write_text(f"{_FINAL_REPORT_LINE}\n", encoding="utf-8")
    out = tmp_path / "runs"

    exit_code = cli.main(
        [
            "run",
            str(probe_dir),
            "--stub-script",
            str(script),
            "--context-tokens",
            "1024",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    events = EventLog.read_jsonl(out / "flag-probe" / "stub-s0" / "events.jsonl")
    dropped, _ = _compaction_dropped_seqs(events)
    assert dropped == {0, 1}


def test_default_window_holds_the_same_traffic_without_compaction(
    tmp_path: Path,
) -> None:
    probe_dir = _make_seeding_probe(tmp_path, "wide-probe", None)
    script = tmp_path / "final.jsonl"
    script.write_text(f"{_FINAL_REPORT_LINE}\n", encoding="utf-8")
    out = tmp_path / "runs"

    exit_code = cli.main(["run", str(probe_dir), "--stub-script", str(script), "--out", str(out)])

    assert exit_code == 0
    events = EventLog.read_jsonl(out / "wide-probe" / "stub-s0" / "events.jsonl")
    assert not [event for event in events if event.type == "compaction"]


def test_manifest_context_tokens_below_floor_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_dir = _make_seeding_probe(tmp_path, "tiny-probe", 512)

    exit_code = cli.main(["run", str(probe_dir), "--out", str(tmp_path / "runs")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "probe rejected" in captured.err
