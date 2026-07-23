# Upgrade v1.21.0 to v1.22.0

## 1. Extract the patch

Extract the patch-only ZIP over the existing project root.

After extraction, remove the superseded root patch note:

```powershell
.\scripts\apply_phase7_cleanup.ps1
```

## 2. Rebuild the app images once

Phase 7 changes `requirements/app.txt` to align Requests dependencies. Rebuild
only the app runtime and thin app image:

```powershell
.\scripts\docker\build-app-runtime.ps1
.\scripts\docker\build-app.ps1
```

Do not rebuild the VLM runtime or VLM thin image.

## 3. Recreate only receipt-app

```powershell
docker compose `
  -f docker-compose.yml `
  -f docker-compose.dev.yml `
  up -d --no-build --force-recreate --no-deps receipt-app
```

## 4. Validate dependencies and tests

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/check_dependency_compatibility.py

docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/run_tests.py
```

## 5. Check readiness

```powershell
Invoke-RestMethod http://localhost:7860/api/readiness | ConvertTo-Json -Depth 10
```
