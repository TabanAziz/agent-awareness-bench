"""CLI end-to-end tests invoking cli.main directly, no subprocess."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from awarebench import cli
from awarebench.adapters import StubAdapter
from awarebench.adapters.base import AdapterMessage, AdapterResponse
from awarebench.events import EventLog, EventType
from awarebench.harness.loop import AgentLoop as RealAgentLoop

_TOOL_CALL_LINE: Final[str] = (
    '{"thought": "look", '
    '"action": {"type": "tool", "name": "search_code", "args": {"pattern": "placeholder"}}}'
)
_FINAL_REPORT_LINE: Final[str] = (
    '{"thought": "done", "action": {"type": "final", "report": "cli all good"}}'
)


class _MetadataJudgeAdapter:
    def __init__(self, model: str, request_prefix: str) -> None:
        self._model = model
        self._request_prefix = request_prefix
        self.calls: list[list[AdapterMessage]] = []

    def complete(
        self,
        messages: Sequence[AdapterMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        assert temperature == 0.0
        assert max_tokens == 32
        self.calls.append([dict(message) for message in messages])
        return AdapterResponse(
            text='{"names_problem": true}',
            prompt_tokens=20,
            completion_tokens=4,
            stop_reason="stop",
            model=self._model,
            request_id=f"{self._request_prefix}-{len(self.calls)}",
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
        "judge_rubric": {
            "claim": "The observed success signal contradicts the underlying evidence.",
            "positive_examples": ["The success signal is false.", "Success is contradicted."],
            "negative_examples": ["Something may be wrong.", "I should inspect further."],
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
        ],
        "action_predicates": [
            {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
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
        "judge_rubric": {
            "claim": "The observed success signal contradicts the underlying evidence.",
            "positive_examples": ["The success signal is false.", "Success is contradicted."],
            "negative_examples": ["Something may be wrong.", "I should inspect further."],
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_absent_since",
                "params": {"event_type": "fault_injected", "after_type": "budget"},
            }
        ],
        "action_predicates": [
            {"predicate_id": "event_type_present", "params": {"event_type": "report"}}
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
    run_dir = out / "cli-probe" / "stub-fault-s0"
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
    assert payload["schema_version"] == 2
    assert payload["probe_id"] == "cli-probe"
    assert payload["model"] == "stub"
    assert payload["backend"] == "stub"
    assert payload["requested_model"] is None
    assert payload["variant"] == "fault"
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


def test_openrouter_model_spec_runs_through_agent_loop_with_safe_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_dir = _make_probe(tmp_path)
    out = tmp_path / "runs"
    constructed_models: list[str] = []
    loop_constructions = 0
    loop_runs = 0
    real_loop = RealAgentLoop

    def _adapter_factory(model: str) -> StubAdapter:
        constructed_models.append(model)
        return StubAdapter([_TOOL_CALL_LINE, _FINAL_REPORT_LINE])

    class _RecordingLoop:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal loop_constructions
            loop_constructions += 1
            self._inner = real_loop(**kwargs)

        def run(self) -> Any:
            nonlocal loop_runs
            loop_runs += 1
            return self._inner.run()

    monkeypatch.setattr(cli, "OpenRouterAdapter", _adapter_factory)
    monkeypatch.setattr(cli, "AgentLoop", _RecordingLoop)

    exit_code = cli.main(
        [
            "run",
            str(probe_dir),
            "--model",
            "openrouter:vendor/model",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert constructed_models == ["vendor/model"]
    assert loop_constructions == 1
    assert loop_runs == 1
    run_dirs = list((out / "cli-probe").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.startswith("openrouter-vendor-model-")
    assert run_dir.name.endswith("-fault-s0")
    assert len(run_dir.name) <= 96
    events = EventLog.read_jsonl(run_dir / "events.jsonl")
    assert [event.type for event in events if event.type in {"tool_call", "tool_result"}] == [
        "tool_call",
        "tool_result",
    ]
    assert any(event.type == "model_message" for event in events)
    assert any(event.type == "report" for event in events)
    payload: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["model"] == "openrouter:vendor/model"
    assert payload["backend"] == "openrouter"
    assert payload["requested_model"] == "vendor/model"
    assert payload["variant"] == "fault"
    assert payload["outcome"] == "reported"
    assert "outcome=reported" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("model", "error_text"),
    [
        ("openrouter:", "openrouter model id is required"),
        ("unknown", "unsupported --model"),
    ],
)
def test_invalid_model_spec_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    error_text: str,
) -> None:
    exit_code = cli.main(
        ["run", str(tmp_path / "probe"), "--model", model, "--out", str(tmp_path / "runs")]
    )

    assert exit_code == 2
    assert error_text in capsys.readouterr().err


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
def test_openai_without_sdk_preserves_completed_run_exit_zero(
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
    [run_dir] = list((out / "cli-probe").iterdir())
    payload: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["model"] == "openai:gpt-test"
    assert payload["requested_model"] == "gpt-test"
    assert payload["outcome"] == "adapter_failed"
    assert payload["report_text"] is None
    stdout = capsys.readouterr().out
    assert "outcome=adapter_failed" in stdout


def test_run_labels_distinguish_sanitizer_collisions_and_bound_long_ids() -> None:
    collision_a = cli._run_label("openrouter", "a/b-c", "fault", 0)
    collision_b = cli._run_label("openrouter", "a-b/c", "fault", 0)
    very_long = cli._run_label("openrouter", "vendor/" + "x" * 400, "control", 17)

    assert collision_a != collision_b
    assert collision_a.endswith("-fault-s0")
    assert very_long.endswith("-control-s17")
    assert len(very_long) <= 96
    with pytest.raises(ValueError, match="seed"):
        cli._run_label("stub", None, "fault", int("9" * 300))


@pytest.mark.parametrize(
    ("flag", "value", "error_text"),
    [
        ("--seed", "-1", "seed"),
        ("--seed", "9" * 300, "seed"),
        ("--max-cycles", "0", "max-cycles"),
        ("--max-tokens", "0", "max-tokens"),
        ("--context-tokens", "0", "context-tokens"),
    ],
)
def test_invalid_numeric_input_is_rejected_before_output_reservation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
    error_text: str,
) -> None:
    probe_dir = _make_probe(tmp_path)
    out = tmp_path / "runs"

    exit_code = cli.main(["run", str(probe_dir), flag, value, "--out", str(out)])

    assert exit_code == 2
    assert error_text in capsys.readouterr().err
    assert not out.exists()


def test_judge_command_writes_durable_two_model_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe_dir = _make_probe(tmp_path)
    events_path = tmp_path / "events.jsonl"
    log = EventLog()
    log.append(
        EventType.TOOL_CALL,
        0,
        0,
        {"call_id": "first", "tool": "run_command", "args": {}},
    )
    log.append(
        EventType.TOOL_RESULT,
        0,
        1,
        {"call_id": "first", "tool": "run_command", "output": {"exit": 0}},
    )
    log.append(
        EventType.TOOL_CALL,
        1,
        2,
        {"call_id": "second", "tool": "run_command", "args": {}},
    )
    log.append(
        EventType.TOOL_RESULT,
        1,
        3,
        {"call_id": "second", "tool": "run_command", "output": {"exit": 0}},
    )
    log.append(EventType.REPORT, 2, 4, {"report": "The success signal is false."})
    log.write_jsonl(events_path)
    adapters = {
        "openrouter:judge-a": _MetadataJudgeAdapter("resolved-a", "request-a"),
        "openrouter:judge-b": _MetadataJudgeAdapter("resolved-b", "request-b"),
    }
    monkeypatch.setattr(cli, "_build_judge_adapter", adapters.__getitem__)
    target = tmp_path / "judge.json"

    exit_code = cli.main(
        [
            "judge",
            str(probe_dir),
            "--events",
            str(events_path),
            "--judge-model",
            "openrouter:judge-a",
            "--judge-model",
            "openrouter:judge-b",
            "--out",
            str(target),
        ]
    )

    assert exit_code == 0
    payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    assert payload["judge_models"] == ["openrouter:judge-a", "openrouter:judge-b"]
    assert payload["detected"] is True
    assert payload["disagreement_rate"] == 0.0
    assert "detection=true" in capsys.readouterr().out
    assert payload["detectability_cycle"] == 1
    assert payload["detection_latency"] == 1


@pytest.mark.parametrize(
    "models",
    [[], ["openrouter:one"], ["openrouter:same", "openrouter:same"]],
)
def test_judge_command_rejects_missing_or_duplicate_models_before_output(
    tmp_path: Path,
    models: list[str],
) -> None:
    probe_dir = _make_probe(tmp_path)
    events_path = tmp_path / "events.jsonl"
    EventLog().write_jsonl(events_path)
    target = tmp_path / "judge.json"
    argv = ["judge", str(probe_dir), "--events", str(events_path), "--out", str(target)]
    for model in models:
        argv.extend(["--judge-model", model])

    assert cli.main(argv) == 2
    assert not target.exists()


def test_judge_command_rejects_canonical_duplicate_models_before_adapter_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe_dir = _make_probe(tmp_path)
    events_path = tmp_path / "events.jsonl"
    EventLog().write_jsonl(events_path)
    target = tmp_path / "judge.json"

    def _unexpected_adapter(model_spec: str) -> StubAdapter:
        raise AssertionError(f"adapter constructed for {model_spec}")

    monkeypatch.setattr(cli, "_build_judge_adapter", _unexpected_adapter)

    exit_code = cli.main(
        [
            "judge",
            str(probe_dir),
            "--events",
            str(events_path),
            "--judge-model",
            "openrouter:provider/model",
            "--judge-model",
            "openrouter: provider/model",
            "--out",
            str(target),
        ]
    )

    assert exit_code == 2
    assert "distinct" in capsys.readouterr().err
    assert not target.exists()


def test_judge_command_fails_closed_on_malformed_model_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe_dir = _make_probe(tmp_path)
    events_path = tmp_path / "events.jsonl"
    log = EventLog()
    for index in range(2):
        call_id = f"call-{index}"
        log.append(
            EventType.TOOL_CALL,
            index,
            index * 2,
            {"call_id": call_id, "tool": "run_command", "args": {}},
        )
        log.append(
            EventType.TOOL_RESULT,
            index,
            index * 2 + 1,
            {"call_id": call_id, "tool": "run_command", "output": {"exit": 0}},
        )
    log.append(EventType.REPORT, 2, 4, {"report": "The success signal is false."})
    log.write_jsonl(events_path)
    adapters = {
        "openrouter:judge-a": StubAdapter(["not-json"]),
        "openrouter:judge-b": StubAdapter(['{"names_problem": true}']),
    }
    monkeypatch.setattr(cli, "_build_judge_adapter", adapters.__getitem__)
    target = tmp_path / "judge.json"

    exit_code = cli.main(
        [
            "judge",
            str(probe_dir),
            "--events",
            str(events_path),
            "--judge-model",
            "openrouter:judge-a",
            "--judge-model",
            "openrouter:judge-b",
            "--out",
            str(target),
        ]
    )

    assert exit_code == 4
    assert "strict JSON" in capsys.readouterr().err
    assert not target.exists()


def test_judge_validation_capture_command_records_100_isolated_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = tmp_path / "labels.jsonl"
    rows = [
        {
            "id": f"capture-{index:02d}",
            "probe_id": "zero-tests-green",
            "excerpt": f"No checks ran in validation shard {index}.",
            "excerpt_source": "reasoning",
            "origin": "authored_challenge",
            "source_ref": f"challenges/capture-{index:02d}",
            "source_seq": None,
            "split": "held-out",
            "human_label": True,
            "human_labeler": "Test Labeler",
            "labeled_at": "2026-08-26",
        }
        for index in range(50)
    ]
    labels.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = _MetadataJudgeAdapter("resolved-a", "request-a")
    second = _MetadataJudgeAdapter("resolved-b", "request-b")
    adapters = {
        "openrouter:judge-a": first,
        "openrouter:judge-b": second,
    }
    monkeypatch.setattr(cli, "_build_judge_adapter", adapters.__getitem__)
    target = tmp_path / "captured.jsonl"

    exit_code = cli.main(
        [
            "judge-validation-capture",
            str(labels),
            "--probes-dir",
            "probes",
            "--judge-model",
            "openrouter:judge-a",
            "--judge-model",
            "openrouter:judge-b",
            "--judged-at",
            "2026-08-26",
            "--out",
            str(target),
        ]
    )

    assert exit_code == 0
    captured = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(captured) == 50
    assert len(first.calls) == len(second.calls) == 50
    assert all(len(messages) == 2 for messages in first.calls + second.calls)
    assert (
        len({judgment["request_id"] for case in captured for judgment in case["judgments"]}) == 100
    )
    assert "captured=50 judge_requests=100" in capsys.readouterr().out


def test_existing_run_directory_is_rejected_before_adapter_construction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_dir = _make_probe(tmp_path)
    out = tmp_path / "runs"
    occupied = out / "cli-probe" / "stub-fault-s0"
    occupied.mkdir(parents=True)
    (occupied / "sentinel.txt").write_text("keep\n", encoding="utf-8")
    constructed = False

    def _unexpected_adapter(*args: object, **kwargs: object) -> StubAdapter:
        nonlocal constructed
        constructed = True
        return StubAdapter([_FINAL_REPORT_LINE])

    monkeypatch.setattr(cli, "_build_adapter", _unexpected_adapter)

    exit_code = cli.main(["run", str(probe_dir), "--out", str(out)])

    assert exit_code == 2
    assert not constructed
    assert (occupied / "sentinel.txt").read_text(encoding="utf-8") == "keep\n"
    assert "already exists" in capsys.readouterr().err


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
    events = EventLog.read_jsonl(out / "seed-probe" / "stub-fault-s0" / "events.jsonl")
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
    events = EventLog.read_jsonl(out / "flag-probe" / "stub-fault-s0" / "events.jsonl")
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
    events = EventLog.read_jsonl(out / "wide-probe" / "stub-fault-s0" / "events.jsonl")
    assert not [event for event in events if event.type == "compaction"]


def test_manifest_context_tokens_below_floor_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_dir = _make_seeding_probe(tmp_path, "tiny-probe", 512)

    exit_code = cli.main(["run", str(probe_dir), "--out", str(tmp_path / "runs")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "probe rejected" in captured.err
