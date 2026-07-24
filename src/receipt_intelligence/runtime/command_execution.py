"""Safe parsing helpers for trusted, server-configured subprocess commands.

Commands remain configurable for local deployments, but they are always
executed with ``shell=False``. Shell operators are rejected because pipes,
redirection, chaining, and variable expansion are intentionally unsupported.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping

_SHELL_CONTROL_TOKENS = {
    "|",
    "||",
    "&",
    "&&",
    ";",
    ">",
    ">>",
    "<",
    "<<",
    "2>",
    "2>>",
}


def split_command(command: str) -> list[str]:
    """Parse a configured command into an argument vector.

    The returned vector is suitable for ``subprocess.run(..., shell=False)``.
    The command must name an executable directly; shell syntax is not accepted.
    """

    text = str(command or "").strip()
    if not text:
        raise ValueError("Configured command is empty.")

    if os.name == "nt":
        argv = shlex.split(text, posix=False)
        argv = [
            argument[1:-1]
            if (
                len(argument) >= 2
                and argument[0] == argument[-1]
                and argument[0] in {"\"", "'"}
            )
            else argument
            for argument in argv
        ]
    else:
        argv = shlex.split(text, posix=True)
    if not argv:
        raise ValueError("Configured command contains no executable.")

    unsupported = [token for token in argv if token in _SHELL_CONTROL_TOKENS]
    if unsupported:
        raise ValueError(
            "Shell operators are not supported in configured commands: "
            + ", ".join(sorted(set(unsupported)))
        )
    return argv


def expand_command_template(command_template: str, values: Mapping[str, str]) -> list[str]:
    """Expand placeholders inside already-separated command arguments.

    Splitting before expansion keeps a path containing whitespace inside one
    argument instead of allowing it to alter the command structure.
    """

    argv = split_command(command_template)
    try:
        return [argument.format_map(values) for argument in argv]
    except KeyError as exc:
        raise ValueError(f"Unknown command-template placeholder: {exc.args[0]}") from exc
