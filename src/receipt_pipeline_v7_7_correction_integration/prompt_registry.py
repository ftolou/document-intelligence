from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    version: str
    path: Path
    sha256: str
    role: str
    kind: str
    variables: tuple[str, ...]
    schema_path: Path | None = None
    schema_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "version": self.version,
            "path": str(self.path),
            "sha256": self.sha256,
            "role": self.role,
            "kind": self.kind,
            "variables": list(self.variables),
            "schema_path": str(self.schema_path) if self.schema_path else None,
            "schema_sha256": self.schema_sha256,
        }


class PromptRegistry:
    """Loads immutable, version-pinned prompt artifacts and verifies their hashes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prompt manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.registry_version = str(manifest.get("registry_version") or "")
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in manifest.get("prompts", []):
            key = (str(entry["id"]), str(entry["version"]))
            if key in self._entries:
                raise ValueError(f"Duplicate prompt registry entry: {key}")
            self._entries[key] = dict(entry)
        self._text_cache: dict[tuple[str, str], str] = {}
        self._schema_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def record(self, prompt_id: str, version: str) -> PromptRecord:
        key = (prompt_id, version)
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"Unknown prompt version: {prompt_id}@{version}")
        path = (self.root / str(entry["path"])).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Prompt path escapes registry root: {path}")
        return PromptRecord(
            prompt_id=prompt_id,
            version=version,
            path=path,
            sha256=str(entry["sha256"]),
            role=str(entry.get("role") or "user"),
            kind=str(entry.get("kind") or "instruction"),
            variables=tuple(str(value) for value in entry.get("variables", [])),
            schema_path=(
                (self.root / str(entry["schema_path"])).resolve()
                if entry.get("schema_path")
                else None
            ),
            schema_sha256=(str(entry["schema_sha256"]) if entry.get("schema_sha256") else None),
        )

    def load(self, prompt_id: str, version: str) -> str:
        key = (prompt_id, version)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        record = self.record(prompt_id, version)
        content = record.path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != record.sha256:
            raise ValueError(
                f"Prompt hash mismatch for {prompt_id}@{version}: "
                f"expected {record.sha256}, got {digest}"
            )
        text = content.rstrip("\n")
        self._text_cache[key] = text
        return text

    def load_schema(self, prompt_id: str, version: str) -> dict[str, Any]:
        key = (prompt_id, version)
        cached = self._schema_cache.get(key)
        if cached is not None:
            return dict(cached)
        record = self.record(prompt_id, version)
        if record.schema_path is None or record.schema_sha256 is None:
            raise KeyError(f"Prompt has no registered schema: {prompt_id}@{version}")
        if self.root not in record.schema_path.parents:
            raise ValueError(f"Schema path escapes registry root: {record.schema_path}")
        raw = record.schema_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if digest != record.schema_sha256:
            raise ValueError(
                f"Schema hash mismatch for {prompt_id}@{version}: "
                f"expected {record.schema_sha256}, got {digest}"
            )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Schema must be a JSON object: {record.schema_path}")
        self._schema_cache[key] = value
        return dict(value)

    def render(self, prompt_id: str, version: str, **values: Any) -> str:
        record = self.record(prompt_id, version)
        missing = [name for name in record.variables if name not in values]
        if missing:
            raise KeyError(f"Missing template variables for {prompt_id}@{version}: {missing}")
        rendered = Template(self.load(prompt_id, version)).substitute(
            {name: str(value) for name, value in values.items()}
        )
        return rendered
