# VLM adapter architecture

The visual-model subsystem is separated into transport, application policy, and infrastructure adapters.

## Dependency direction

```text
entrypoints/vlm_http
        |
        v
application/vlm
        |
        v
application/ports/vlm
        ^
        |
adapters/vlm
```

The Flask service maps HTTP requests to `VlmAnalysisService`. It does not import PaddleOCR-VL runners or command execution functions directly.

## Adapters

- `PaddlePythonVlmEngine` runs the PaddleOCR-VL Python API.
- `PaddleCliVlmEngine` runs the supported `paddleocr doc_parser` command.
- `RemoteVlmClient` calls the standalone VLM service.
- `TrustedCommandVlmEngine` runs a deployment-owned command template without a shell.

Every adapter implements `VlmEngine` and receives a provider-neutral `VlmRequest`.

## Application policies

`FallbackVlmEngine` owns Python-to-CLI fallback behavior. Concrete adapters do not select or invoke other adapters.

`OptionalVlmEngine` owns enablement checks, missing-image handling, progress events, and persistence of the application-side VLM result.

`VlmAnalysisService` owns image preparation and result enrichment for the standalone service.

## Composition

`receipt_intelligence.composition` is the only module that selects concrete adapters. The receipt application normally uses `RemoteVlmClient`. Local execution can be enabled through trusted deployment configuration.

The standalone service selects its local runner at startup. Request payloads cannot select backends, commands, timeouts, runners, or resize limits.

## Removing obsolete modules

After extracting the patch over an existing repository, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\apply_repository_cleanup.ps1
```

The script removes the former monolithic VLM engine, the former service module, and the temporary compatibility adapter. It is safe to run more than once.
