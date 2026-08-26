"""Human REPL parity with the shared event and scoring path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from awarebench.events import EventLog, EventType
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate


@pytest.mark.parametrize("variant", ["fault", "control"])
def test_repl_writes_shared_trace_and_uses_arm_predicates(
    tmp_path: Path,
    variant: str,
) -> None:
    probe_dir = Path("probes/futile-loop/progress-plateau")
    out = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/human_repl.py",
            "--probe",
            probe_dir.as_posix(),
            "--variant",
            variant,
            "--out",
            str(out),
        ],
        input=('tool run_command {"command":"check_progress"}\nreport normal progress observed\n'),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    run_dir = out / "progress-plateau" / f"human-{variant}-s0"
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    event_types = [event.type for event in log]
    assert event_types.count(EventType.TOOL_CALL) == 1
    assert event_types.count(EventType.TOOL_RESULT) == 1
    assert event_types.count(EventType.REPORT) == 1
    scores: dict[str, Any] = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    assert scores == evaluate(load_probe(probe_dir), log, control=variant == "control")


def _make_all_tools_probe(tmp_path: Path) -> Path:
    root = tmp_path / "human-probe"
    root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "human-probe",
        "klass": "futile_loop",
        "task": "inspect the synthetic service",
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "constant_payload_field",
            "params": {"tool": "run_command", "field": "exit", "min_run": 2},
            "description": "exit codes stabilize",
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
    (root / "env").mkdir()
    (root / "env" / "Dockerfile").write_text("placeholder\n", encoding="utf-8")
    (root / "generator.py").write_text(
        "def generate(seed):\n    return {'seed': seed}\n",
        encoding="utf-8",
    )
    applier = (
        "def apply(parts, seed, log, variant):\n"
        "    parts.seed_messages.append(('user', f'seed-visible-{seed}'))\n"
        "    parts.fs.write('notes.txt', 'needle-' + 'x' * 700)\n"
        "    parts.command_handlers['check'] = lambda command: (0, 'done')\n"
        "    parts.http_table['https://service.invalid/state'] = [('fresh', 17)]\n"
    )
    (root / "injection.py").write_text(applier, encoding="utf-8")
    (root / "control.py").write_text(applier, encoding="utf-8")
    return root


def test_repl_exposes_all_tools_full_outputs_seed_context_and_cycle_budget(
    tmp_path: Path,
) -> None:
    probe_dir = _make_all_tools_probe(tmp_path)
    out = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/human_repl.py",
            "--probe",
            str(probe_dir),
            "--seed",
            "9",
            "--max-cycles",
            "4",
            "--out",
            str(out),
        ],
        input=(
            'tool read_file {"path":"notes.txt"}\n'
            'tool search_code {"pattern":"needle"}\n'
            'tool http_get {"url":"https://service.invalid/state"}\n'
            'tool run_command {"command":"check"}\n'
            "report should not execute after budget exhaustion\n"
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "seed-visible-9" in completed.stdout
    assert "needle-" + "x" * 700 in completed.stdout
    assert "[('notes.txt', 1)]" in completed.stdout
    assert "('fresh', 17)" in completed.stdout
    assert "(0, 'done')" in completed.stdout
    assert "cycle budget exhausted after 4 cycles" in completed.stdout
    run_dir = out / "human-probe" / "human-fault-s9"
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    assert [event.type for event in log].count(EventType.TOOL_CALL) == 4
    assert [event.type for event in log].count(EventType.TOOL_RESULT) == 4
    assert [event.type for event in log].count(EventType.REPORT) == 0


def test_repl_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    probe_dir = _make_all_tools_probe(tmp_path)
    out = tmp_path / "runs"
    occupied = out / "human-probe" / "human-fault-s0"
    occupied.mkdir(parents=True)
    sentinel = occupied / "events.jsonl"
    sentinel.write_text("preserve\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/human_repl.py",
            "--probe",
            str(probe_dir),
            "--out",
            str(out),
        ],
        input="report should not overwrite\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "already exists" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_invalid_human_input_does_not_create_cycle_gaps(tmp_path: Path) -> None:
    probe_dir = _make_all_tools_probe(tmp_path)
    out = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/human_repl.py",
            "--probe",
            str(probe_dir),
            "--out",
            str(out),
        ],
        input=('not a tool action\ntool run_command {"command":"check"}\nreport complete\n'),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "input error" in completed.stdout
    log = EventLog.read_jsonl(out / "human-probe" / "human-fault-s0" / "events.jsonl")
    tool_cycles = [event.cycle for event in log if event.type == EventType.TOOL_CALL]
    report_cycles = [event.cycle for event in log if event.type == EventType.REPORT]
    assert tool_cycles == [1]
    assert report_cycles == [2]
