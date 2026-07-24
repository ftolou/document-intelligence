#!/usr/bin/env python3
"""Helpers to unload/reload Ollama so VLM can temporarily use the GPU.

This is optional and best-effort only. It supports two modes:
  - api: call the Ollama HTTP API and ask it to unload or warm-load a model
  - command: run trusted executable commands without a shell
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.request
from typing import Any

from receipt_intelligence.runtime.command_execution import split_command


def _json_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url.rstrip("/"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if not text.strip():
        return {"ok": True, "raw": ""}
    try:
        return json.loads(text)
    except Exception:
        return {"ok": True, "raw": text}


def _run_command(cmd: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    argv = split_command(cmd)
    proc = subprocess.run(
        argv,
        shell=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "command": shlex.join(argv),
        "argv": argv,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "duration_seconds": round(time.perf_counter() - started, 2),
    }


def unload_ollama(
    *,
    ollama_url: str,
    model: str,
    timeout_seconds: float = 60.0,
    mode: str = "api",
    unload_command: str = "",
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if mode == "command" and unload_command.strip():
            res = _run_command(unload_command, timeout_seconds)
        else:
            # Ask Ollama to unload this model from VRAM. Empty prompt is fine here.
            data = _json_post(
                f"{ollama_url.rstrip('/')}/api/generate",
                {
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
                timeout_seconds,
            )
            res = {"status": "ok", "mode": "api", "response": data}
        if wait_seconds and wait_seconds > 0:
            time.sleep(wait_seconds)
        res.setdefault("mode", mode)
        res["duration_seconds"] = round(time.perf_counter() - started, 2)
        return res
    except Exception as exc:
        return {
            "status": "error",
            "mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


def reload_ollama(
    *,
    ollama_url: str,
    model: str,
    keep_alive: str = "10m",
    timeout_seconds: float = 120.0,
    mode: str = "api",
    start_command: str = "",
    warmup_prompt: str = "ok",
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if mode == "command" and start_command.strip():
            res = _run_command(start_command, timeout_seconds)
        else:
            # Warm-load the model again so the correction pass does not pay cold-start cost.
            data = _json_post(
                f"{ollama_url.rstrip('/')}/api/generate",
                {
                    "model": model,
                    "prompt": warmup_prompt or "ok",
                    "stream": False,
                    "keep_alive": keep_alive,
                },
                timeout_seconds,
            )
            res = {"status": "ok", "mode": "api", "response": data}
        if wait_seconds and wait_seconds > 0:
            time.sleep(wait_seconds)
        res.setdefault("mode", mode)
        res["duration_seconds"] = round(time.perf_counter() - started, 2)
        return res
    except Exception as exc:
        return {
            "status": "error",
            "mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
