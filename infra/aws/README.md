# BESSER SmartGen AWS foundation

This Terraform stack creates the durable and isolated execution foundation for
SmartGen while leaving the existing web editor controller on EC2:

- encrypted PostgreSQL/RDS run state;
- private KMS-encrypted S3 artifacts and checkpoints;
- an encrypted SQS job queue and dead-letter queue;
- an immutable ECR worker repository;
- a supervised, private ECS/Fargate worker service with one fresh task per run;
- least-privilege worker and controller IAM policies;
- queue-backlog autoscaling plus CloudWatch queue-age and dead-letter alarms.

It intentionally does **not** mutate the existing VPC, DNS, EC2 instance, or
security group. Supply the existing VPC, two private subnets, and controller
security group through `terraform.tfvars`. The worker subnets need outbound
access to the configured LLM providers and AWS APIs, normally through NAT or
VPC endpoints plus an egress proxy.

## Apply

Terraform state contains the generated database password and database URL.
Local state is therefore disabled. Before `terraform init`, create a private
versioned S3 state bucket with public access blocked and default KMS encryption,
then copy `backend.hcl.example` to the ignored `backend.hcl`. The backend uses
encrypted state and native S3 lock files; administrators need scoped access to
the state object and its `.tflock` object.

1. Copy `terraform.tfvars.example` to `terraform.tfvars`, replace every
   infrastructure placeholder, and copy `backend.hcl.example` to `backend.hcl`.
2. On the first release only, run `./deploy-smartgen-worker.sh bootstrap` from
   the workspace root. Commit the generated `.terraform.lock.hcl` before any
   production apply.
3. Run `./deploy-smartgen-worker.sh push`, then
   `./deploy-smartgen-worker.sh plan`. The plan is read-only and uses the exact
   digest behind the immutable `git-<full-sha>` tag.
4. Run `./deploy-smartgen-worker.sh apply`, review its fresh saved plan, and
   type the requested revision to approve it. The script applies only that
   plan, waits for ECS steady state, and verifies the task definition uses the
   expected digest. CI must set `SMARTGEN_WORKER_AUTO_APPROVE=1` explicitly.
5. Re-run `./deploy-smartgen-worker.sh verify` at any time for a read-only
   ECR/Terraform/ECS freshness check.
6. Add the Terraform outputs to the backend production environment:

   ```text
   BESSER_SMARTGEN_STATE_MODE=production
   BESSER_SMARTGEN_DATABASE_URL=<Secrets Manager value>
   BESSER_SMARTGEN_S3_BUCKET=<smartgen_artifact_bucket>
   BESSER_SMARTGEN_SQS_QUEUE_URL=<smartgen_queue_url>
   BESSER_SMARTGEN_AWS_REGION=eu-north-1
   BESSER_SMARTGEN_KMS_KEY_ID=<smartgen_kms_key_arn>
   BESSER_SMARTGEN_S3_KMS_KEY_ID=<smartgen_kms_key_arn>
   BESSER_SMARTGEN_DURABLE_JOBS=true
   ```

   The generated database secret uses `sslmode=verify-full` and the
   `/opt/aws/rds/global-bundle.pem` trust bundle installed in both BESSER
   production images. Do not weaken this to `require` or omit `sslrootcert`.

Never place a provider API key in Terraform state, task definitions, SQS
messages, or container environment files. The controller encrypts per-run BYOK
material with KMS and workers delete it at terminal state.

Always release the worker before the EC2 backend. `deploy.sh backend` and
`deploy.sh all` run the same read-only verifier before building or pushing any
Compose image, so a missing, stale, mutable, or unstable worker blocks the
release without changing EC2. Terraform requires `worker_image` to be an ECR
`repo@sha256:...` reference; tag-based and `bootstrap` task definitions are
rejected.

The controller only persists the run and sends its opaque run ID to SQS; it has
no `ecs:RunTask` or `iam:PassRole` permission. The ECS service continuously
long-polls the queue. A task handles one delivered message, cleans its workspace,
and exits; ECS replaces it with a fresh task. Visible backlog scales the service
out. Scale-in occurs only after both visible and in-flight message counts are
zero, so an invisible in-progress message cannot be terminated by autoscaling.

Paid jobs also use the ECS agent task-protection endpoint. The worker enables
protection after validating and hydrating a received job but before marking
`execution_started` or entering the paid runner. Lease heartbeats refresh that
protection, and cleanup disables it before the one-job task exits. Production
fails closed when `ECS_AGENT_URI` is missing or an agent update times out, so a
worker never starts a new paid call without confirmed protection. The task role
has only `ecs:GetTaskProtection` and `ecs:UpdateTaskProtection` for tasks in this
cluster. `worker_task_protection_minutes` and
`worker_task_protection_timeout_seconds` bound protection and agent calls;
`worker_stop_timeout_seconds` gives checkpoint/cleanup up to the Fargate maximum
120 seconds after SIGTERM. Protected tasks may intentionally delay a deployment
until their paid job completes or the bounded protection expires.

The controller and worker keep arbitrary shell tools disabled by default.
`worker_allow_shell_tools=true` is only acceptable for a trusted single-tenant
deployment: approval controls intent, but the current worker role still has
shared S3, SQS, KMS, and database access. Multi-tenant shell execution needs a
credentialless sidecar or a per-run scoped IAM role before it can be enabled.
When enabled, every command still requires durable user approval and
`worker_approval_timeout_seconds` controls how long the task waits. The image
declares a pre-owned `/workspace` volume so ECS copies its non-root permissions
into the ephemeral bind mount. Temporary files and tool caches stay there while
the task's root filesystem remains read-only.
