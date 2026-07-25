# Security boundaries

These controls define the security boundaries enforced by the current deployment.

## Enforced boundaries

- Receipt-upload form fields cannot override Ollama or VLM service URLs.
- Receipt-upload form fields cannot provide VLM or Ollama lifecycle commands.
- The VLM HTTP service uses backend, runner, timeout, resize, and command values only from deployment configuration.
- The VLM HTTP service accepts shared image paths only below `VLM_ALLOWED_INPUT_ROOTS`.
- VLM result identifiers are sanitized before they are used as directory names.
- Configured commands execute as argument vectors with `shell=False`; shell operators are rejected.
- The VLM container is available to Compose services through the internal network but port `7870` is not published on the host.

## Configuration

`VLM_ALLOWED_INPUT_ROOTS` is an operating-system path-list. The Docker Compose default is:

```text
/app/var
```

Configured commands must name an executable directly. Pipes, redirects, command chaining, and other shell syntax are unsupported. A valid template is:

```text
python -m my_vlm_wrapper --image "{image}" --out "{output_json}"
```

## Verification

Run:

```bash
python scripts/check_security_boundaries.py
python scripts/run_tests.py tests/unit/security
```

The static check is also part of `scripts/run_quality_checks.py`.
