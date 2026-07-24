"""Trusted command adapter for deployment-specific VLM wrappers."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest
from receipt_intelligence.runtime.command_execution import expand_command_template
from receipt_intelligence.runtime.json_values import jsonable


def run_trusted_command(
    command_template: str, image_path: Path, output_json_path: Path, timeout_seconds: float
) -> dict[str, Any]:
    started = time.perf_counter()
    cmd: list[str] = []
    try:
        cmd = expand_command_template(
            command_template,
            {
                "image": str(image_path),
                "output_json": str(output_json_path),
                "output": str(output_json_path),
            },
        )
        proc = subprocess.run(
            cmd,
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=str(image_path.parent),
        )
        raw_result: Any = None
        if output_json_path.exists():
            try:
                raw_result = json.loads(output_json_path.read_text(encoding="utf-8-sig"))
            except Exception:
                raw_result = output_json_path.read_text(encoding="utf-8", errors="replace")
        elif proc.stdout.strip():
            try:
                raw_result = json.loads(proc.stdout)
            except Exception:
                raw_result = proc.stdout
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "backend": "command",
            "command": shlex.join(cmd),
            "argv": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "raw_result": jsonable(raw_result),
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "error",
            "backend": "command",
            "command": shlex.join(cmd) if cmd else command_template,
            "argv": cmd,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


class TrustedCommandVlmEngine(VlmEngine):
    """Execute a trusted server-side command template without a shell."""

    def __init__(self, command_template: str) -> None:
        self.command_template = command_template

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        if request.image_path is None:
            return {
                "status": "skipped",
                "backend": "command",
                "message": "No image path was provided.",
            }
        output_json = request.result_dir / f"{request.run_id}_command_vlm_output.json"
        return run_trusted_command(
            self.command_template,
            request.image_path,
            output_json,
            request.timeout_seconds,
        )


__all__ = ["TrustedCommandVlmEngine", "run_trusted_command"]
