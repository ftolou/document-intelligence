from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from receipt_intelligence.adapters.vlm import trusted_command
from receipt_intelligence.runtime.command_execution import split_command
from receipt_intelligence.services import ollama_control

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"


def test_source_tree_does_not_enable_shell_execution() -> None:
    violations: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def test_shell_operators_are_rejected_from_configured_commands() -> None:
    with pytest.raises(ValueError, match="Shell operators"):
        split_command("python worker.py ; touch unexpected-file")


def test_ollama_command_runner_uses_argument_vector_without_shell(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(ollama_control.subprocess, "run", fake_run)

    result = ollama_control._run_command('python -m worker --reason "gpu handoff"', 5.0)

    assert result["status"] == "ok"
    assert captured["argv"] == ["python", "-m", "worker", "--reason", "gpu handoff"]
    assert captured["shell"] is False


def test_vlm_command_template_expands_paths_as_single_arguments(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    image = tmp_path / "receipt image.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "result output.json"

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(trusted_command.subprocess, "run", fake_run)

    result = trusted_command.run_trusted_command(
        'python wrapper.py --image "{image}" --out "{output_json}"',
        image,
        output,
        5.0,
    )

    assert result["status"] == "ok"
    assert captured["argv"] == [
        "python",
        "wrapper.py",
        "--image",
        str(image),
        "--out",
        str(output),
    ]
    assert captured["shell"] is False
