"""PaddleOCR-VL CLI execution owned entirely by the VLM service."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def run_paddle_cli(
    image_path: Path,
    result_dir: Path,
    timeout_seconds: float,
    *,
    device: str,
    engine: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = result_dir / "paddleocr_vl_cli_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_args = os.getenv("VLM_CLI_EXTRA_ARGS", "").strip()

    command = [
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
        command.extend(["--engine", engine])
    if extra_args:
        command.extend(shlex.split(extra_args))

    try:
        environment = os.environ.copy()
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        environment.setdefault("FLAGS_selected_gpus", "0")
        environment.setdefault("PADDLE_DEVICE", device)
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=environment,
            shell=False,
        )

        files: list[dict[str, Any]] = []
        produced_files: list[str] = []
        for candidate in sorted(out_dir.rglob("*")):
            if not candidate.is_file():
                continue
            produced_files.append(str(candidate))
            if candidate.suffix.lower() not in {".json", ".md", ".txt", ".html"}:
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content: Any = text
            if candidate.suffix.lower() == ".json":
                try:
                    content = json.loads(text)
                except Exception:
                    pass
            files.append(
                {
                    "path": str(candidate),
                    "name": candidate.name,
                    "suffix": candidate.suffix.lower(),
                    "content": content,
                }
            )

        status = "ok" if process.returncode == 0 and (files or produced_files) else "error"
        error = None
        if status != "ok":
            reasons: list[str] = []
            if process.returncode != 0:
                reasons.append(f"CLI exited with return code {process.returncode}")
            if not files and not produced_files:
                reasons.append("CLI produced no output files")
            if process.stderr.strip():
                reasons.append(process.stderr.strip()[-1500:])
            elif process.stdout.strip():
                reasons.append(process.stdout.strip()[-1500:])
            error = " | ".join(reasons) or "PaddleOCR-VL CLI failed without stderr."

        return {
            "status": status,
            "backend": "paddleocr_vl_cli",
            "device": device,
            "engine": engine,
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": process.returncode,
            "error": error,
            "stdout_tail": process.stdout[-6000:],
            "stderr_tail": process.stderr[-6000:],
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
            "command": " ".join(shlex.quote(part) for part in command),
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


__all__ = ["run_paddle_cli"]
