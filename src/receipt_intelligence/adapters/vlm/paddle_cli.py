"""Local PaddleOCR-VL command-line adapter."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest


def run_paddle_cli(image_path: Path, result_dir: Path, timeout_seconds: float) -> dict[str, Any]:
    """Run PaddleOCR-VL via the doc_parser CLI.

    V14.7.9 locks the service to the standalone-tested route on this machine:
      paddleocr doc_parser -i IMAGE --save_path OUT --device gpu:0 --engine transformers

    The CLI writes JSON/MD/DOCX/images; this function collects JSON/MD/TXT/HTML
    content and also records all produced file paths for diagnostics.
    """
    started = time.perf_counter()
    out_dir = result_dir / "paddleocr_vl_cli_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = os.getenv("VLM_DEVICE", "gpu:0").strip() or "gpu:0"
    engine = os.getenv("VLM_ENGINE", "transformers").strip()
    extra_args = os.getenv("VLM_CLI_EXTRA_ARGS", "").strip()

    cmd = [
        "paddleocr",
        "doc_parser",
        "-i",
        str(image_path),
        "--save_path",
        str(out_dir),
        "--device",
        device,
    ]
    if engine:
        cmd.extend(["--engine", engine])
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    try:
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("FLAGS_selected_gpus", "0")
        env.setdefault("PADDLE_DEVICE", device)
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_seconds, env=env)
        files: list[dict[str, Any]] = []
        produced_files: list[str] = []
        for fp in sorted(out_dir.rglob("*")):
            if not fp.is_file():
                continue
            produced_files.append(str(fp))
            if fp.suffix.lower() not in {".json", ".md", ".txt", ".html"}:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content: Any = text
            if fp.suffix.lower() == ".json":
                try:
                    content = json.loads(text)
                except Exception:
                    pass
            files.append(
                {"path": str(fp), "name": fp.name, "suffix": fp.suffix.lower(), "content": content}
            )

        status = "ok" if proc.returncode == 0 and (files or produced_files) else "error"
        error = None
        if status != "ok":
            reason = []
            if proc.returncode != 0:
                reason.append(f"CLI exited with return code {proc.returncode}")
            if not files and not produced_files:
                reason.append("CLI produced no output files")
            if proc.stderr.strip():
                reason.append(proc.stderr.strip()[-1500:])
            elif proc.stdout.strip():
                reason.append(proc.stdout.strip()[-1500:])
            error = " | ".join(reason) or "PaddleOCR-VL CLI failed without stderr."

        return {
            "status": status,
            "backend": "paddleocr_vl_cli",
            "device": device,
            "engine": engine,
            "command": " ".join(shlex.quote(x) for x in cmd),
            "returncode": proc.returncode,
            "error": error,
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
            "output_dir": str(out_dir),
            "produced_files": produced_files,
            "raw_result": {"files": files},
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "error",
            "backend": "paddleocr_vl_cli",
            "device": device,
            "engine": engine,
            "command": " ".join(shlex.quote(x) for x in cmd),
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


class PaddleCliVlmEngine(VlmEngine):
    """Execute PaddleOCR-VL through the supported doc_parser CLI."""

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        if request.image_path is None:
            return {
                "status": "skipped",
                "backend": "paddleocr_vl_cli",
                "message": "No image path was provided.",
            }
        return run_paddle_cli(
            request.image_path,
            request.result_dir,
            request.timeout_seconds,
        )


__all__ = ["PaddleCliVlmEngine", "run_paddle_cli"]
