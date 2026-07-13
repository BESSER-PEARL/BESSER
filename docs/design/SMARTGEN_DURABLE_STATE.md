# SmartGen durable state foundation

The package
`besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state`
provides the persistence boundary required to detach SmartGen work from an HTTP/SSE
connection. It backs the additive durable run API while the legacy synchronous
route remains available for compatibility.

## Public API

```python
from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    DurableStateConfig,
    JobMessage,
    RunRecord,
    build_durable_state,
)

config = DurableStateConfig.from_env()
foundation = build_durable_state(config)
await foundation.initialize()

await foundation.state.create_run(RunRecord(...))
await foundation.state.append_event(run_id, "phase", {"phase": "generate"})
await foundation.queue.enqueue(JobMessage(run_id, owner_id, payload))
```

`DurableStateFoundation` exposes:

- `state`: run records, optimistic updates, ordered events, replay cursors,
  idempotency claims, fenced leases, and expiring quota reservations.
- `artifacts`: generated-file upload, integrity-checked download, deletion, and
  short-lived download URLs.
- `checkpoints`: atomic latest-checkpoint storage per owner and run.
- `queue`: enqueue, long-poll receive, acknowledge, release, and visibility
  extension while a worker is active.
- `initialize()` and `close()` lifecycle methods.

The local adapter set is SQLite + filesystem + an in-memory queue. The production
adapter set is PostgreSQL + encrypted S3 + encrypted SQS.

## Configuration

| Variable | Local | Production | Default |
| --- | --- | --- | --- |
| `BESSER_SMARTGEN_STATE_MODE` | optional | required operationally | `local` |
| `BESSER_SMARTGEN_DURABLE_JOBS` | optional | optional | enabled in production mode |
| `BESSER_SMARTGEN_SQLITE_PATH` | optional | ignored | OS temp data directory |
| `BESSER_SMARTGEN_STORAGE_DIR` | optional | ignored | OS temp data directory |
| `BESSER_SMARTGEN_DATABASE_URL` | ignored | required with `sslmode=verify-full` and absolute `sslrootcert` | none |
| `BESSER_SMARTGEN_S3_BUCKET` | ignored | required | none |
| `BESSER_SMARTGEN_S3_PREFIX` | ignored | optional | `smartgen` |
| `BESSER_SMARTGEN_S3_KMS_KEY_ID` | ignored | optional | S3-managed encryption |
| `BESSER_SMARTGEN_SQS_QUEUE_URL` | ignored | required | none |
| `BESSER_SMARTGEN_AWS_REGION` | ignored | optional | AWS SDK resolution |
| `BESSER_SMARTGEN_IDEMPOTENCY_TTL_SECONDS` | optional | optional | `86400` |
| `BESSER_SMARTGEN_LEASE_TTL_SECONDS` | optional | optional | `60` |
| `BESSER_SMARTGEN_MAX_CONCURRENT_RUNS_PER_OWNER` | optional | optional | `2` |
| `BESSER_SMARTGEN_MAX_STARTS_PER_HOUR` | optional | optional | `10` |
| `BESSER_SMARTGEN_MAX_STORAGE_BYTES_PER_OWNER` | optional | optional | `1073741824` |
| `BESSER_SMARTGEN_MAX_DURABLE_REQUEST_BYTES` | optional | optional | `5242880` |
| `BESSER_SMARTGEN_EVENT_PAGE_SIZE` | optional | optional | `200` (max `1000`) |
| `BESSER_SMARTGEN_ARTIFACT_URL_TTL_SECONDS` | optional | optional | `300` (max `3600`) |

Production adapter dependencies are installed by the backend requirements:

```bash
pip install -r besser/utilities/web_modeling_editor/backend/requirements.txt
```

The adapters still load these packages lazily. When production mode is selected,
missing configuration, `psycopg`, `boto3`, an
unreachable database/bucket/queue, or an unencrypted SQS queue raises during
construction/initialization. The factory never falls back to SQLite, filesystem,
or memory.

## Run and event semantics

- `RunRecord.owner_id` is immutable. API routes should use `get_owned_run()` and
  return the same not-found response for missing and foreign runs.
- Run updates require the current `version`. A stale version receives
  `OptimisticLockError` instead of overwriting newer state.
- Events receive a monotonically increasing per-run sequence inside the same
  database transaction as their insert.
- `ReplayCursor` binds the last sequence to its run. A reconnect reads events
  strictly after that cursor and can map it to SSE `Last-Event-ID`.
- Worker leases retain a monotonically increasing fencing token.
  `update_run_fenced()` and `append_event_fenced()` verify the worker identity,
  token, and unexpired lease in the same lock/database transaction as the write.
  A stale worker receives `LeaseLostError`.
- Production workers activate the ECS agent task-protection endpoint before
  the first paid runner step, refresh it from the same heartbeat loop that
  renews the lease and SQS visibility, and remove it during cleanup. Agent URI,
  response, or timeout failures stop and retry rather than running unprotected.
- A quota reservation is atomic per owner/resource. Do not release start-rate
  reservations; let their TTL implement the time window. Release concurrent-run
  and storage staging reservations when the corresponding resource is freed.

## Idempotent admission

The API should generate a proposed run ID, hash a canonical request with SHA-256,
then claim `(owner_id, Idempotency-Key)` before charging or enqueueing. A repeated
key with the same hash returns the original run ID; a different hash raises
`IdempotencyConflictError`.

There is a small, intentional recovery window between claiming and creating a run.
If the process stops there, a retry receives the stored proposed run ID and should
create that same missing run before continuing. Concurrent creators then converge
through `RecordAlreadyExistsError`. Reserve start/concurrency quota only after the
idempotency claim so HTTP retries do not consume quota twice.

Admission after the claim runs in a shielded runtime task. A browser or proxy
disconnect cannot cancel database creation, queue publication, or worker
dispatch halfway through. Replaying an already-created run never dispatches
again. In production the controller only publishes to SQS; a managed worker
service long-polls the queue independently.

## HTTP contract

- `POST /besser_api/smart-gen/runs` requires `Idempotency-Key` and returns `202`
  with `{run_id,status}` plus a `Location` header.
- `GET /besser_api/smart-gen/runs` and `GET /besser_api/smart-gen/runs/{run_id}`
  return only owner-safe summaries. Stored requests and secret envelopes are
  never serialized.
- `GET /besser_api/smart-gen/runs/{run_id}/events` replays stored payloads as
  SSE. The integer event sequence is the SSE ID; `Last-Event-ID` resumes
  strictly after it.
- Owned cancellation, approval resolution, and artifact download live beneath
  the same run URL. Missing and foreign runs intentionally share `404`.
- `GET /besser_api/smart-gen/config` exposes `features.durable_jobs` so the
  editor can choose the durable client without guessing deployment state.

Production BYOK encryption additionally requires `BESSER_SMARTGEN_KMS_KEY_ID`.
ECS cluster, task, subnet, and security-group settings belong to deployment
configuration for the managed worker service, not the controller process.

## Artifact and checkpoint security

- Owner IDs are SHA-256-derived in blob keys; raw account identifiers do not appear
  in filesystem paths or S3 object names.
- Local writes use temporary files plus `os.replace`; checkpoints include an
  integrity sidecar. S3 objects carry SHA-256 metadata and always request SSE-S3 or
  SSE-KMS encryption.
- S3 download URLs are capped at one hour. Authorization must happen before a URL
  is created; a presigned URL is then a temporary bearer capability.
- Queue serialization rejects common plaintext secret fields such as `api_key`,
  `access_token`, and `password`. A BYOK credential must travel as a short-lived
  encrypted envelope reference/ciphertext, never plaintext SQS JSON.
- Configure S3 lifecycle deletion, versioning policy, public-access blocking, and
  an SQS dead-letter queue in infrastructure. The adapters do not weaken missing
  infrastructure policy in application code.

## Production operations

1. Install the production adapter requirements.
2. Provision PostgreSQL with TLS, backups, and a least-privilege application role.
   Use `sslmode=verify-full&sslrootcert=/opt/aws/rds/global-bundle.pem`.
   Both production images install the AWS global RDS CA bundle at that path.
3. Provision a private S3 bucket with lifecycle expiry and KMS if required.
4. Provision an encrypted SQS queue with a dead-letter queue and a visibility
   timeout longer than the lease heartbeat interval.
5. Set `BESSER_SMARTGEN_STATE_MODE=production` and all required values.
6. Call `await foundation.initialize()` before accepting traffic. Initialization
   creates the version-one SQL tables and verifies S3/SQS access.
7. Run code execution in a separate one-job worker/container. The API process
   should only persist state, publish events, and dispatch jobs.
