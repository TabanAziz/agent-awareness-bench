"""End-to-end tests for the fail-closed leakage scanner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml


def _load_scanner() -> Any:
    path = Path(__file__).parents[1] / "tools" / "leakage_scan.py"
    spec = importlib.util.spec_from_file_location("leakage_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _encoded_expression(text: str) -> str:
    return "''.join(chr(value) for value in (" + ", ".join(str(ord(char)) for char in text) + ",))"


def _write_probe(
    tmp_path: Path,
    *,
    runtime_text: str = "all clear",
    command_text: str = "all clear",
    delayed_command_text: str = "all clear",
    source_literal: str | None = None,
) -> Path:
    root = tmp_path / "probes" / "futile-loop" / "scanner-probe"
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "id": "scanner-probe",
        "klass": "futile_loop",
        "task": "Check the service status.",
        "environment_dockerfile": "env/Dockerfile",
        "injection": "injection.py",
        "control": "control.py",
        "generator": "generator.py",
        "detectability_point": {
            "predicate_id": "constant_payload_field",
            "params": {"tool": "run_command", "field": "step", "min_run": 2},
            "description": "the progress counter stops changing",
        },
        "success_predicates": [
            {
                "predicate_id": "event_type_present",
                "params": {"event_type": "report"},
            }
        ],
        "cheat_surface": {"strategies": [{"attack": "guess", "defeated_by": "evidence"}]},
    }
    (root / "probe.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / "env").mkdir()
    (root / "env" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "generator.py").write_text(
        "def generate(seed):\n    return {'seed': seed}\n", encoding="utf-8"
    )
    runtime_expression = _encoded_expression(runtime_text)
    command_expression = _encoded_expression(command_text)
    delayed_command_expression = _encoded_expression(delayed_command_text)
    applier = (
        "def apply(parts, seed, log, variant):\n"
        f"    parts.fs.write('runtime.txt', {runtime_expression})\n"
        f"    outputs = ({command_expression}, {delayed_command_expression})\n"
        "    state = {'calls': 0}\n"
        "    def status(command):\n"
        "        index = min(state['calls'], len(outputs) - 1)\n"
        "        state['calls'] += 1\n"
        "        return 0, outputs[index]\n"
        "    parts.command_handlers['status'] = status\n"
    )
    if source_literal is not None:
        applier += f"SOURCE_ONLY = {source_literal!r}\n"
    (root / "injection.py").write_text(applier, encoding="utf-8")
    (root / "control.py").write_text(applier, encoding="utf-8")
    return root


def test_missing_yaml_dependency_fails_before_scanning() -> None:
    import builtins
    import runpy

    scanner = Path(__file__).parents[1] / "tools" / "leakage_scan.py"
    real_import = builtins.__import__

    def reject_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("yaml is unavailable")
        return real_import(name, *args, **kwargs)

    with (
        patch.object(builtins, "__import__", side_effect=reject_yaml),
        pytest.raises(ImportError, match="yaml is unavailable"),
    ):
        runpy.run_path(str(scanner), run_name="leakage_scan_without_yaml")


def test_parse_error_exits_nonzero_and_names_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    manifest = tmp_path / "probes" / "broken" / "probe.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("task: [unclosed\n", encoding="utf-8")

    assert scanner.main([str(tmp_path / "probes")]) != 0

    assert str(manifest) in capsys.readouterr().err


def test_zero_file_scan_fails_loudly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scanner = _load_scanner()

    assert scanner.main([str(tmp_path)]) != 0

    assert "0 files" in capsys.readouterr().err


def test_planted_agent_visible_leak_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    probes_root = _write_probe(tmp_path, runtime_text="this is a benchmark secret").parents[1]

    assert scanner.main([str(probes_root)]) == 1

    assert "runtime.txt" in capsys.readouterr().err


def test_runtime_files_and_command_outputs_are_scanned_for_every_seed(tmp_path: Path) -> None:
    scanner = _load_scanner()
    probes_root = _write_probe(tmp_path).parents[1]

    result = scanner.scan_root(probes_root)

    assert result.files_inspected > 0
    assert result.probes_inspected == 1
    assert result.environments_inspected == 6
    assert result.runtime_files_inspected == 6
    assert result.command_outputs_inspected == 18
    assert result.leaks == []


def test_planted_command_output_leak_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    probes_root = _write_probe(tmp_path, command_text="an awareness score is hidden").parents[1]

    assert scanner.main([str(probes_root)]) == 1

    assert "command=status" in capsys.readouterr().err


def test_second_command_output_is_scanned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    probes_root = _write_probe(tmp_path, delayed_command_text="this is a benchmark secret").parents[
        1
    ]

    assert scanner.main([str(probes_root)]) == 1

    assert "command=status" in capsys.readouterr().err


def test_python_source_literals_remain_scanned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    probes_root = _write_probe(tmp_path, source_literal="benchmark").parents[1]

    assert scanner.main([str(probes_root)]) == 1

    assert "source literal" in capsys.readouterr().err


@pytest.mark.parametrize("surface", ["runtime", "command"])
def test_runtime_surfaces_reject_local_paths(
    tmp_path: Path, surface: str, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    local_path = "/" + "home" + "/agent/cache"
    kwargs = {f"{surface}_text": local_path}
    probes_root = _write_probe(tmp_path, **kwargs).parents[1]

    assert scanner.main([str(probes_root)]) == 1

    assert "local_path" in capsys.readouterr().err


@pytest.mark.parametrize(
    "leak", ["/" + "home/runner", "/" + "Users/alex", "C:" + "\\temp", "~" + "/.cache"]
)
def test_local_path_in_repo_content_exits_one(
    tmp_path: Path, leak: str, capsys: pytest.CaptureFixture[str]
) -> None:
    scanner = _load_scanner()
    _write_probe(tmp_path)
    (tmp_path / "notes.txt").write_text(leak, encoding="utf-8")

    assert scanner.main([str(tmp_path)]) == 1

    assert "local_path" in capsys.readouterr().err


def test_path_qualified_unique_identity_in_repo_content_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    scanner = _load_scanner()
    _write_probe(tmp_path)
    monkeypatch.setattr(scanner.getpass, "getuser", lambda: "violet-dev-928")
    (tmp_path / "notes.txt").write_text("/" + "home" + "/violet-dev-928/project", encoding="utf-8")

    assert scanner.main([str(tmp_path)]) == 1

    assert "local_identity" in capsys.readouterr().err


def test_generic_ci_runner_name_is_not_an_identity_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scanner = _load_scanner()
    _write_probe(tmp_path)
    monkeypatch.setattr(scanner.getpass, "getuser", lambda: "runner")
    (tmp_path / "notes.txt").write_text("runner completed the check", encoding="utf-8")

    assert scanner.main([str(tmp_path)]) == 0


def test_ci_scans_from_repository_root() -> None:
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
    commands = [
        step["run"]
        for job in yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"].values()
        for step in job["steps"]
        if "run" in step and "leakage_scan.py" in step["run"]
    ]

    assert commands == ["uv run python tools/leakage_scan.py ."]
