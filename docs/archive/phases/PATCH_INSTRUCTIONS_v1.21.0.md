# v1.21.0 Phase 6 patch instructions

This patch removes compatibility files that ZIP extraction cannot delete by
itself. Apply it only over the v1.20.0 Phase 5 project.

## 1. Stop the services

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml stop receipt-app receipt-vlm
```

## 2. Extract the patch over the project

Extract all patch files into the repository root and allow replacement of
existing files.

## 3. Run the cutover and cleanup

```powershell
.\scripts\apply_phase6_cleanup.ps1
```

The script copies any remaining legacy data into `var/`, creates missing job
manifests, writes a migration report, creates a ZIP backup beside the project,
and then removes obsolete folders and compatibility files. It stops on file
conflicts rather than overwriting them silently.

## 4. Recreate both containers

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d `
  --no-build --force-recreate receipt-app receipt-vlm
```

The images do not need to be rebuilt. Container recreation is required because
legacy bind mounts were removed from Compose.

## 5. Validate

```powershell
python scripts/verify_runtime_layout.py

docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/run_tests.py
```
