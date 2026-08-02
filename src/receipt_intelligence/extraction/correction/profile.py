from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    kind: str
    prompt_id: str
    prompt_version: str
    max_attempts: int
    max_patches: int


@dataclass(frozen=True)
class CorrectionProfile:
    profile_version: str
    automatic_patching: bool
    retain_accepted_partial_corrections: bool
    max_rounds: int
    routes: dict[str, tuple[str, ...]]
    strategies: dict[str, StrategyConfig]
    source_path: Path

    def strategy_chain(self, validation_code: str) -> tuple[StrategyConfig, ...]:
        strategy_ids = self.routes.get(validation_code, self.routes.get("*", ()))
        return tuple(self.strategies[strategy_id] for strategy_id in strategy_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "automatic_patching": self.automatic_patching,
            "retain_accepted_partial_corrections": (
                self.retain_accepted_partial_corrections
            ),
            "max_rounds": self.max_rounds,
            "source_path": str(self.source_path),
            "routes": {key: list(value) for key, value in self.routes.items()},
            "strategies": {
                key: {
                    "kind": value.kind,
                    "prompt_id": value.prompt_id,
                    "prompt_version": value.prompt_version,
                    "max_attempts": value.max_attempts,
                    "max_patches": value.max_patches,
                }
                for key, value in self.strategies.items()
            },
        }


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Correction profile field {key!r} must be boolean")
    return value


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Correction profile field {key!r} must be >= 1")
    return value


def load_correction_profile(path: Path) -> CorrectionProfile:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Correction profile must be a JSON object")

    profile_version = str(payload.get("profile_version") or "").strip()
    if not profile_version:
        raise ValueError("Correction profile requires profile_version")

    raw_strategies = payload.get("strategies")
    if not isinstance(raw_strategies, dict) or not raw_strategies:
        raise ValueError("Correction profile requires nonempty strategies")

    strategies: dict[str, StrategyConfig] = {}
    for strategy_id, raw in raw_strategies.items():
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise ValueError("Strategy IDs must be nonempty strings")
        if not isinstance(raw, dict):
            raise ValueError(f"Strategy {strategy_id!r} must be an object")
        kind = str(raw.get("kind") or "").strip()
        if kind != "source_evidence":
            raise ValueError(
                f"Unsupported strategy kind for {strategy_id}: {kind}. "
                "Only source_evidence strategies are permitted."
            )
        prompt_id = str(raw.get("prompt_id") or "").strip()
        prompt_version = str(raw.get("prompt_version") or "").strip()
        if not prompt_id or not prompt_version:
            raise ValueError(f"Strategy {strategy_id!r} requires prompt binding")
        max_attempts = raw.get("max_attempts")
        max_patches = raw.get("max_patches")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError(f"Strategy {strategy_id!r} max_attempts must be >= 1")
        if isinstance(max_patches, bool) or not isinstance(max_patches, int) or max_patches < 1:
            raise ValueError(f"Strategy {strategy_id!r} max_patches must be >= 1")
        strategies[strategy_id] = StrategyConfig(
            strategy_id=strategy_id,
            kind=kind,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            max_attempts=max_attempts,
            max_patches=max_patches,
        )

    raw_routes = payload.get("routes")
    if not isinstance(raw_routes, dict):
        raise ValueError("Correction profile routes must be an object")
    routes: dict[str, tuple[str, ...]] = {}
    for validation_code, raw_chain in raw_routes.items():
        if not isinstance(validation_code, str) or not validation_code:
            raise ValueError("Route keys must be nonempty strings")
        if not isinstance(raw_chain, list) or not raw_chain:
            raise ValueError(f"Route {validation_code!r} must be a nonempty list")
        chain = tuple(str(value) for value in raw_chain)
        unknown = [value for value in chain if value not in strategies]
        if unknown:
            raise ValueError(
                f"Route {validation_code!r} references unknown strategies: {unknown}"
            )
        if len(set(chain)) != len(chain):
            raise ValueError(f"Route {validation_code!r} contains duplicates")
        routes[validation_code] = chain

    return CorrectionProfile(
        profile_version=profile_version,
        automatic_patching=_require_bool(payload, "automatic_patching"),
        retain_accepted_partial_corrections=_require_bool(
            payload, "retain_accepted_partial_corrections"
        ),
        max_rounds=_require_positive_int(payload, "max_rounds"),
        routes=routes,
        strategies=strategies,
        source_path=resolved,
    )
