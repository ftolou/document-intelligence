"""Versioned and hash-verified prompt artifact registry.

The active prompt loader remains available for backward compatibility. New extraction stages can
adopt this registry one prompt family at a time without changing existing prompt behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PromptRegistryError(RuntimeError):
    """Base error raised by prompt artifact loading and verification."""


class PromptNotFoundError(PromptRegistryError):
    pass


class PromptIntegrityError(PromptRegistryError):
    pass


class PromptRenderError(PromptRegistryError):
    pass


@dataclass(frozen=True, slots=True)
class PromptReference:
    prompt_id: str
    version: str

    def __post_init__(self) -> None:
        prompt_id = str(self.prompt_id or "").strip()
        version = str(self.version or "").strip()
        if not prompt_id or not version:
            raise ValueError("PromptReference requires non-empty prompt_id and version.")
        object.__setattr__(self, "prompt_id", prompt_id)
        object.__setattr__(self, "version", version)


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    reference: PromptReference
    template_path: Path
    template_sha256: str
    required_variables: tuple[str, ...] = ()
    schema_path: Path | None = None
    schema_sha256: str | None = None
    metadata: dict[str, Any] | None = None


class PromptRegistry:
    """Loads immutable prompt artifacts declared in ``manifest.json``."""

    def __init__(self, root: Path, *, manifest_name: str = "manifest.json") -> None:
        self._root = Path(root)
        self._manifest_path = self._root / manifest_name
        self._entries = self._load_manifest()

    @property
    def root(self) -> Path:
        return self._root

    def references(self) -> tuple[PromptReference, ...]:
        return tuple(sorted(self._entries, key=lambda ref: (ref.prompt_id, ref.version)))

    def resolve(self, reference: PromptReference) -> PromptArtifact:
        try:
            artifact = self._entries[reference]
        except KeyError as exc:
            raise PromptNotFoundError(
                f"Prompt not registered: {reference.prompt_id}@{reference.version}"
            ) from exc
        self._verify_artifact(artifact)
        return artifact

    def read_template(self, reference: PromptReference) -> str:
        artifact = self.resolve(reference)
        return artifact.template_path.read_text(encoding="utf-8")

    def load_schema(self, reference: PromptReference) -> dict[str, Any] | None:
        artifact = self.resolve(reference)
        if artifact.schema_path is None:
            return None
        payload = json.loads(artifact.schema_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PromptRegistryError(
                f"Prompt schema must be a JSON object: {artifact.schema_path}"
            )
        return payload

    def render(self, reference: PromptReference, **values: Any) -> str:
        artifact = self.resolve(reference)
        missing = [name for name in artifact.required_variables if name not in values]
        if missing:
            raise PromptRenderError(
                f"Missing prompt variables for {reference.prompt_id}@{reference.version}: "
                + ", ".join(sorted(missing))
            )
        text = artifact.template_path.read_text(encoding="utf-8")
        if not artifact.required_variables:
            return text

        alternatives = "|".join(
            re.escape(name) for name in sorted(artifact.required_variables, key=len, reverse=True)
        )
        placeholder_pattern = re.compile(
            r"\{\{(?P<mustache>" + alternatives + r")\}\}"
            r"|\$\{(?P<braced_dollar>" + alternatives + r")\}"
            r"|\$(?P<plain_dollar>" + alternatives + r")(?![A-Za-z0-9_])"
        )

        sentinels: dict[str, str] = {}
        for index, name in enumerate(artifact.required_variables):
            sentinel = f"\x1fPROMPT_VARIABLE_{index}\x1f"
            while sentinel in text:
                sentinel += "_"
            sentinels[name] = sentinel

        def replace_with_sentinel(match: re.Match[str]) -> str:
            name = (
                match.group("mustache")
                or match.group("braced_dollar")
                or match.group("plain_dollar")
            )
            return sentinels[name]

        text = placeholder_pattern.sub(replace_with_sentinel, text)
        unresolved_matches = list(placeholder_pattern.finditer(text))
        if unresolved_matches:
            unresolved = sorted(
                {
                    match.group("mustache")
                    or match.group("braced_dollar")
                    or match.group("plain_dollar")
                    for match in unresolved_matches
                }
            )
            raise PromptRenderError("Unresolved prompt variables: " + ", ".join(unresolved))

        for name, sentinel in sentinels.items():
            replacement = "" if values[name] is None else str(values[name])
            text = text.replace(sentinel, replacement)
        return text

    def _load_manifest(self) -> dict[PromptReference, PromptArtifact]:
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PromptRegistryError(f"Prompt manifest not found: {self._manifest_path}") from exc
        if not isinstance(payload, dict):
            raise PromptRegistryError("Prompt manifest root must be a JSON object.")
        entries = payload.get("prompts")
        if not isinstance(entries, list):
            raise PromptRegistryError("Prompt manifest 'prompts' must be an array.")

        artifacts: dict[PromptReference, PromptArtifact] = {}
        for raw in entries:
            if not isinstance(raw, dict):
                raise PromptRegistryError("Every prompt manifest entry must be an object.")
            reference = PromptReference(
                prompt_id=str(raw.get("id") or ""),
                version=str(raw.get("version") or ""),
            )
            if reference in artifacts:
                raise PromptRegistryError(
                    f"Duplicate prompt registration: {reference.prompt_id}@{reference.version}"
                )
            template_path = self._resolve_relative_path(raw.get("template"), "template")
            schema_path = (
                self._resolve_relative_path(raw.get("schema"), "schema")
                if raw.get("schema") is not None
                else None
            )
            required_variables = raw.get("required_variables") or []
            if not isinstance(required_variables, list) or not all(
                isinstance(value, str) and value.strip() for value in required_variables
            ):
                raise PromptRegistryError("required_variables must be an array of strings.")
            artifacts[reference] = PromptArtifact(
                reference=reference,
                template_path=template_path,
                template_sha256=self._require_hash(raw.get("sha256"), "sha256"),
                required_variables=tuple(value.strip() for value in required_variables),
                schema_path=schema_path,
                schema_sha256=(
                    self._require_hash(raw.get("schema_sha256"), "schema_sha256")
                    if schema_path is not None
                    else None
                ),
                metadata=dict(raw.get("metadata") or {}),
            )
        return artifacts

    def _verify_artifact(self, artifact: PromptArtifact) -> None:
        self._verify_text_hash(artifact.template_path, artifact.template_sha256)
        if artifact.schema_path is not None and artifact.schema_sha256 is not None:
            self._verify_text_hash(artifact.schema_path, artifact.schema_sha256)

    def _resolve_relative_path(self, value: Any, label: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise PromptRegistryError(f"Prompt manifest {label} path must not be empty.")
        path = (self._root / raw).resolve()
        root = self._root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PromptRegistryError(f"Prompt {label} path escapes registry root: {raw}") from exc
        return path

    @staticmethod
    def _require_hash(value: Any, label: str) -> str:
        digest = str(value or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PromptRegistryError(f"{label} must be a 64-character SHA-256 digest.")
        return digest

    @staticmethod
    def _verify_text_hash(path: Path, expected: str) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromptIntegrityError(f"Prompt artifact not found: {path}") from exc
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected:
            raise PromptIntegrityError(
                f"Prompt artifact hash mismatch for {path}: expected {expected}, got {actual}"
            )


def default_prompt_registry() -> PromptRegistry:
    return PromptRegistry(Path(__file__).resolve().parent)


__all__ = [
    "PromptArtifact",
    "PromptIntegrityError",
    "PromptNotFoundError",
    "PromptReference",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptRenderError",
    "default_prompt_registry",
]
