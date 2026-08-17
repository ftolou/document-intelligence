# Background job execution

Receipt extraction and batch processing are submitted through the application
`JobDispatcher` port. HTTP routes and application use cases do not create
threads or executors.

## Runtime model

The default adapter is a bounded process-level `ThreadPoolJobDispatcher`:

- `JOB_WORKER_MAX_WORKERS` controls active workers;
- `JOB_QUEUE_CAPACITY` bounds waiting work;
- one worker is the safe default for local Paddle, Qwen, and Gemma execution;
- every submitted task is described by a serializable dispatch payload in
  `job_status.json`;
- state transitions and timestamps are persisted before and after execution;
- a per-job filesystem claim prevents duplicate work across app processes that
  share the same `var/jobs` volume;
- a managed lease heartbeat keeps active claims current;
- queued or interrupted jobs are discovered from persisted state and retried
  after an expired claim.

The implementation is local and process-level, but the use-case contract does
not depend on threads. A durable external queue can replace the adapter without
changing routes or job submission use cases.

## Persisted lifecycle

New jobs use these states:

```text
queued -> running -> completed
                  -> failed
```

Each status document includes:

```text
created_at
queued_at
started_at
finished_at
attempt_count
error
```

Older `done` and `error` states remain readable by the frontend for existing job
artifacts.

## Failure behavior

Processor exceptions propagate to the dispatcher. The dispatcher records the
exception type, message, traceback, terminal state, and completion timestamp.
A full bounded queue returns HTTP 503 with `error_code=job_queue_full`; the
rejected job is persisted as failed rather than remaining indefinitely queued.

## Shutdown and recovery

The composition boundary owns the dispatcher and registers graceful process
shutdown. Active executor work is allowed to finish during a normal shutdown.
The maintenance coordinator is process-managed and renews job claims while work
is active.

On startup, `JOB_RECOVER_PENDING=1` scans persisted queued or running jobs. A
running job owned by another live process retains its claim and is not executed
twice. If its process disappears, the lease expires and a surviving dispatcher
can claim and retry it.

## Configuration

```env
JOB_WORKER_MAX_WORKERS=1
JOB_QUEUE_CAPACITY=32
JOB_CLAIM_LEASE_SECONDS=120
JOB_MAINTENANCE_INTERVAL_SECONDS=10
JOB_RECOVER_PENDING=1
```

Keep the claim lease at least three times the maintenance interval. Increase the
worker count only when the transcription and structured-extraction resource model supports concurrent
jobs safely.
