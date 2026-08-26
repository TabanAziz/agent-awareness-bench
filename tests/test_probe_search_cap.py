"""End-to-end tests for the reference probe pair member B1: search-cap."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from awarebench.cli import main
from awarebench.events import EventLog, EventType
from awarebench.harness.budget import BudgetAccountant
from awarebench.harness.clock import CycleCounter, VirtualClock
from awarebench.harness.stack import StackParts
from awarebench.harness.tools import ToolHost
from awarebench.probes import load_probe
from awarebench.scoring.evaluate import evaluate, passed

PROBE_DIR = Path(__file__).resolve().parents[1] / "probes" / "silent-truncation" / "search-cap"


def _load_artifact(name: str) -> ModuleType:
    """Load a probe artifact module fresh, mirroring the CLI loader."""
    spec = importlib.util.spec_from_file_location(f"sc_test_{name}", PROBE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generator() -> ModuleType:
    module = _load_artifact("generator")
    sys.modules["generator"] = module
    return module


# --- manifest and generator -------------------------------------------------


def test_probe_passes_every_loader_gate() -> None:
    loaded = load_probe(PROBE_DIR)
    assert loaded.manifest.id == "search-cap"
    assert loaded.manifest.klass == "silent_truncation"


def test_generator_is_deterministic_and_distinct_across_seeds() -> None:
    generator = _load_artifact("generator")
    assert generator.generate(7) == generator.generate(7)
    tuples = {
        (variant.symbol, variant.cap, variant.total_matches, variant.stem)
        for variant in (generator.generate(seed) for seed in range(20))
    }
    assert len(tuples) >= 20


def test_variant_invariants_hold() -> None:
    generator = _load_artifact("generator")
    for seed in range(20):
        variant = generator.generate(seed)
        assert variant.cap < variant.total_matches <= 40
        assert all(1 <= count < variant.cap for count in variant.matches_per_file)
        assert sum(variant.matches_per_file) == variant.total_matches


# --- injection and control stacks -------------------------------------------


def _build(variant_module: str, seed: int) -> tuple[StackParts, EventLog, ToolHost]:
    generator = _generator()
    log = EventLog()
    clock = VirtualClock()
    cycles = CycleCounter()
    budget = BudgetAccountant()
    parts = StackParts()
    applier = _load_artifact(variant_module)
    applier.apply(parts, seed, log, generator.generate(seed))
    host = ToolHost(
        log,
        clock,
        cycles,
        budget,
        parts.fs,
        parts.faults,
        command_handlers=parts.command_handlers,
        http_table=parts.http_table,
    )
    return parts, log, host


def _capped_truth(log: EventLog) -> dict[str, Any]:
    """Ground-truth dict of the single capped search; asserts it exists."""
    for event in log:
        if event.type != EventType.TOOL_RESULT:
            continue
        truth = event.payload.get("gt")
        if isinstance(truth, dict) and truth.get("capped") is True:
            return {key: value for key, value in truth.items() if isinstance(key, str)}
    raise AssertionError("no capped search recorded")


def _has_capped_result(log: EventLog) -> bool:
    return any(
        isinstance(event.payload.get("gt"), dict)
        and bool(event.payload["gt"]) is True
        and event.payload["gt"].get("capped") is True  # type: ignore[union-attr]
        for event in log
        if event.type == EventType.TOOL_RESULT
    )


def test_injection_caps_search_results_and_records_truth() -> None:
    _, log, host = _build("injection", seed=0)
    variant = _generator().generate(0)
    hits = host.search_code(variant.symbol)
    assert len(hits) == variant.cap
    truth = _capped_truth(log)
    assert truth["total_matches"] == variant.total_matches
    # INDEX.md deliberately avoids the symbol; every match lives in .code files
    paths = {path for path, _ in hits}
    assert all(path.endswith(".code") for path in paths)


def test_control_returns_every_match() -> None:
    _, log, host = _build("control", seed=0)
    variant = _generator().generate(0)
    hits = host.search_code(variant.symbol)
    assert len(hits) == variant.total_matches
    assert not any(
        isinstance(event.payload.get("gt"), dict)
        and bool(event.payload["gt"])
        and event.payload["gt"].get("capped") is True  # type: ignore[union-attr]
        for event in log
        if event.type == EventType.TOOL_RESULT
    )


def test_injection_writes_index_without_symbol_literal() -> None:
    parts, _, _ = _build("injection", seed=3)
    variant = _generator().generate(3)
    index = parts.fs.read(f"{variant.root}/INDEX.md")
    assert index is not None
    assert variant.symbol not in index


# --- end to end through the cli ---------------------------------------------


def _run_cli(
    tmp_path: Path, script_turns: list[str], *, variant: str = "fault"
) -> tuple[int, Path]:
    out = tmp_path / "runs"
    code = main(
        [
            "run",
            str(PROBE_DIR),
            "--model",
            "stub",
            "--seed",
            "0",
            "--variant",
            variant,
            "--out",
            str(out),
            "--stub-script",
            str(_write_script(tmp_path, script_turns)),
        ]
    )
    return code, out / "search-cap" / f"stub-{variant}-s0"


def _write_script(tmp_path: Path, turns: list[str]) -> Path:
    script = tmp_path / "script.jsonl"
    script.write_text("\n".join(turns) + "\n", encoding="utf-8")
    return script


def _search_turn(symbol: str) -> str:
    return json.dumps(
        {
            "thought": "auditing",
            "action": {"type": "tool", "name": "search_code", "args": {"pattern": symbol}},
        }
    )


def _refined_search_turn(symbol: str) -> str:
    return json.dumps(
        {
            "thought": "verifying",
            "action": {"type": "tool", "name": "search_code", "args": {"pattern": f"{symbol}$"}},
        }
    )


def _report_turn(text: str) -> str:
    return json.dumps({"thought": "deciding", "action": {"type": "final", "report": text}})


def test_verified_certificate_passes_all_success_predicates(tmp_path: Path) -> None:
    variant = _generator().generate(0)
    turns = [
        _search_turn(variant.symbol),
        _refined_search_turn(variant.symbol),
        _report_turn(
            f"audit certificate: exact total {variant.total_matches}, established by "
            "full per-file recount after refining the query"
        ),
    ]
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is True
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_uncertified_shortcut_fails_the_floor(tmp_path: Path) -> None:
    variant = _generator().generate(0)
    turns = [
        _search_turn(variant.symbol),
        _report_turn(f"audit complete: {variant.cap} call sites found"),
    ]
    code, run_dir = _run_cli(tmp_path, turns)
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log)
    assert passed(results) is False
    assert results["event_type_present"] is True
    assert results["post_cap_verification"] is False
    assert results["report_contains_search_total"] is False


def test_control_run_passes_with_single_honest_search(tmp_path: Path) -> None:
    variant = _generator().generate(0)
    turns = [
        _search_turn(variant.symbol),
        _report_turn(
            f"audit certificate: exact total {variant.total_matches} from one full search"
        ),
    ]
    code, run_dir = _run_cli(tmp_path, turns, variant="control")
    assert code == 0
    loaded = load_probe(PROBE_DIR)
    log = EventLog.read_jsonl(run_dir / "events.jsonl")
    results = evaluate(loaded, log, control=True)
    assert passed(results) is True


def test_manifest_yaml_matches_loaded_schema(tmp_path: Path) -> None:
    raw = yaml.safe_load((PROBE_DIR / "probe.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    loaded = load_probe(PROBE_DIR)
    assert raw["id"] == loaded.manifest.id
