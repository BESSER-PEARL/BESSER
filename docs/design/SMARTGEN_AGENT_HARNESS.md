# SmartGen Agent Harness

SmartGen is a model-aware coding agent, not a free-form chat completion. It
combines the editor's deterministic generators with an LLM tool loop and runs
paid work as an owner-bound durable job.

## Execution model

1. The editor exports the complete `ProjectInput`, including the active class,
   GUI, agent, state-machine, object, BPMN, neural-network, or quantum model.
2. The controller authenticates the principal, validates budgets and payload
   size, claims idempotency and quotas, encrypts the BYOK key, creates the run,
   and enqueues only a run reference.
3. An isolated worker acquires a fenced lease, decrypts the key in memory,
   reconstructs the request, and starts `SmartGenerationRunner` with the
   controller-assigned run ID.
4. Phase 1 selects the best compatible deterministic BESSER generator. Phase 2
   fills the model-to-code gaps with workspace-scoped file tools. Phase 3
   validates the generated project and performs bounded fixes.
5. Every event is stored before it is replayed to a browser. A disconnected
   browser reconnects with `Last-Event-ID`; disconnecting never cancels paid
   work. Only the owned cancel endpoint requests cancellation.
6. The worker stores a checksummed artifact and terminal status, releases its
   lease and concurrency reservation, and acknowledges the queue message.

## Trust boundaries

- Application authentication and run ownership are mandatory in production.
- BYOK secrets are envelope-encrypted with a per-run encryption context and
  are never placed in events, logs, queue payloads, Redux, URLs, or durable
  request metadata as plaintext.
- File tools are confined to the generated workspace. Arbitrary commands and
  dependency installation are disabled by default in production. A trusted,
  single-tenant deployment may opt in; every such call then pauses for a
  decision from the authenticated run owner.
- Approval events describe the exact proposed tool and arguments. Approval is
  one-shot; it does not create a permanent command allowlist.
- Command subprocesses receive a minimal environment with secret-like
  variables removed. Denylisted destructive and exfiltration patterns remain
  blocked even after an approval.
- Multi-tenant workers must not enable arbitrary shell execution while the
  subprocess shares a task role or database/network trust with the worker.
  Owner approval does not turn shared cloud credentials into a tenant sandbox.
- Artifacts, checkpoints, events, cancellation, approval, and replay are all
  owner-scoped.

## Durability contract

- PostgreSQL is authoritative for run state, ordered events, idempotency,
  quotas, leases, and approval decisions.
- SQS provides redelivery and a dead-letter queue; workers renew both their
  lease and message visibility while executing.
- In ECS production, a worker must confirm agent task protection before setting
  `execution_started` or entering the paid runner. Heartbeats refresh the
  bounded protection and cleanup removes it. Missing or failed protection is a
  retryable fail-closed condition, never permission to start an unprotected
  paid call.
- S3 stores encrypted, checksummed artifacts and workspace checkpoints.
- Optimistic versions prevent accidental overwrites, while fencing tokens
  prevent a stale worker from committing after its lease is lost.
- Idempotent enqueue returns the original run for the same owner, key, and
  request hash. Reusing a key for a different request is a conflict.

## User experience contract

- The current drawer flow remains direct: the user confirms provider, key, and
  budget, then generation starts. There is no speculative "approve plan"
  screen.
- The run card shows deterministic phases, gap analysis, file/tool activity,
  command approvals, validation, elapsed time, warnings, and the final
  artifact action.
- Network reconnection replays missed events without duplicating tool rows.
- Stop means an explicit durable cancellation request, not merely closing the
  browser stream.

## Deployment modes

`BESSER_SMARTGEN_STATE_MODE=local` uses SQLite and filesystem blobs for tests
and trusted local development. `production` fails closed unless PostgreSQL,
S3, SQS, KMS, and AWS region settings are all present. A supervised ECS
service consumes the queue; the controller never launches a task from an HTTP
request and production never falls back to the in-process worker.

The EC2 controller exposes only the reverse proxy publicly. The database,
queue, bucket, and worker tasks remain private. Worker images run as non-root,
drop Linux capabilities, use a read-only root filesystem, and receive no
controller OAuth or deployment secrets.

## Operational readiness

Before enabling durable jobs in production:

- apply the reviewed infrastructure plan and immutable worker image tag;
- migrate the split service environment files and rotate previously exposed
  provider or OAuth credentials;
- verify GitHub login, cookie-backed WebSocket authentication, run ownership,
  approval, cancellation, replay, artifact download, retry, and DLQ alarms;
- run a paid smoke generation with a low cost/runtime cap; and
- keep the legacy synchronous route disabled for untrusted public traffic.
