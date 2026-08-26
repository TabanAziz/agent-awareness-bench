"""Human REPL parity with the shared event and scoring path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from awarebench.events import EventLog, EventType
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate


def test_control_repl_writes_shared_trace_and_uses_control_predicates(
    tmp_path: Path,
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
            "control",
            "--out",
            str(out),
        ],
        input="check_progress\nreport normal progress observed\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    run_dir = out / "progress-plateau" / "human-control"
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    event_types = [event.type for event in log]
    assert event_types.count(EventType.TOOL_CALL) == 1
    assert event_types.count(EventType.TOOL_RESULT) == 1
    assert event_types.count(EventType.REPORT) == 1
    scores: dict[str, Any] = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    assert scores == evaluate(load_probe(probe_dir), log, control=True)
