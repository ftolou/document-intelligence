# GitHub Upload

## Recommended repository name

```text
document-intelligence-pipeline
```

## Initial push

```powershell
git init
git branch -M main
git add .
git commit -m "Initial document intelligence pipeline release"
git remote add origin https://github.com/<your-user>/document-intelligence-pipeline.git
git push -u origin main
```

## Do not commit

The repository ignores runtime and model data:

```text
var/
model_cache/
.env
*.zip
```

Do not upload receipt photos, generated JSON results, Ollama models, PaddleOCR-VL models, Docker image exports, or local `.env` files.

## After cloning on another machine

Build runtime images once:

```powershell
.\scripts\docker\build-app-runtime.ps1
.\scripts\docker\build-vlm-runtime.ps1
```

Build thin images:

```powershell
.\scripts\docker\build-app.ps1
.\scripts\docker\build-vlm.ps1
```

Start:

```powershell
.\start_windows.ps1
```
