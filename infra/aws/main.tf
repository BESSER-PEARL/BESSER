locals {
  name_prefix        = "besser-smartgen-${var.environment}"
  rds_ca_bundle_path = "/opt/aws/rds/global-bundle.pem"
  alarm_actions = var.alarm_sns_topic_arn == "" ? [] : [
    var.alarm_sns_topic_arn,
  ]
  worker_image = var.worker_image
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_iam_policy_document" "smartgen_kms" {
  statement {
    sid    = "EnableAccountIAMPolicies"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    actions   = ["kms:*"]
    resources = ["*"]
  }

  statement {
    sid    = "AllowCloudWatchLogsEncryption"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["logs.${var.aws_region}.amazonaws.com"]
    }
    actions = [
      "kms:Decrypt*",
      "kms:Describe*",
      "kms:Encrypt*",
      "kms:GenerateDataKey*",
      "kms:ReEncrypt*",
    ]
    resources = ["*"]
    condition {
      test     = "ArnEquals"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values = [
        "arn:${data.aws_partition.current.partition}:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/besser/${var.environment}/smartgen-worker",
      ]
    }
  }
}

resource "aws_kms_key" "smartgen" {
  description             = "BESSER SmartGen artifacts, checkpoints, and short-lived BYOK envelopes"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.smartgen_kms.json
}

resource "aws_kms_alias" "smartgen" {
  name          = "alias/${local.name_prefix}"
  target_key_id = aws_kms_key.smartgen.key_id
}

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${local.name_prefix}-artifacts-"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.smartgen.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-smartgen-data"
    status = "Enabled"

    filter {}

    expiration {
      days = var.artifact_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 1
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_sqs_queue" "dead_letter" {
  name                      = "${local.name_prefix}-dlq"
  message_retention_seconds = 1209600
  kms_master_key_id         = aws_kms_key.smartgen.arn
}

resource "aws_sqs_queue" "jobs" {
  name                       = "${local.name_prefix}-jobs"
  visibility_timeout_seconds = var.worker_visibility_timeout_seconds
  message_retention_seconds  = 1209600
  receive_wait_time_seconds  = 20
  kms_master_key_id          = aws_kms_key.smartgen.arn

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = var.worker_max_attempts
  })
}

resource "aws_db_subnet_group" "smartgen" {
  name       = local.name_prefix
  subnet_ids = var.private_subnet_ids
}

resource "aws_security_group" "worker" {
  name_prefix = "${local.name_prefix}-worker-"
  description = "No-ingress security group for isolated SmartGen workers"
  vpc_id      = var.vpc_id

  egress {
    description = "Temporary outbound access; replace with an allow-listing proxy for strict multi-tenancy"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "database" {
  name_prefix = "${local.name_prefix}-db-"
  description = "PostgreSQL access from the BESSER controller and SmartGen workers"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Controller PostgreSQL"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.controller_security_group_id]
  }

  ingress {
    description     = "Worker PostgreSQL"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.worker.id]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "random_password" "database" {
  length           = 32
  special          = true
  override_special = "-_"
}

resource "random_id" "final_snapshot" {
  byte_length = 4
}

resource "aws_db_instance" "smartgen" {
  identifier = local.name_prefix

  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = var.database_instance_class
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_type           = "gp3"
  storage_encrypted      = true
  kms_key_id             = aws_kms_key.smartgen.arn
  db_name                = var.database_name
  username               = var.database_username
  password               = random_password.database.result
  port                   = 5432
  db_subnet_group_name   = aws_db_subnet_group.smartgen.name
  vpc_security_group_ids = [aws_security_group.database.id]
  publicly_accessible    = false
  multi_az               = var.database_multi_az

  backup_retention_period   = 7
  deletion_protection       = var.database_deletion_protection
  skip_final_snapshot       = false
  final_snapshot_identifier = "${local.name_prefix}-final-${random_id.final_snapshot.hex}"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  auto_minor_version_upgrade      = true
  apply_immediately               = false
}

resource "aws_secretsmanager_secret" "database_url" {
  name_prefix = "${local.name_prefix}/database-url-"
  kms_key_id  = aws_kms_key.smartgen.arn
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id = aws_secretsmanager_secret.database_url.id
  secret_string = format(
    "postgresql://%s:%s@%s:%d/%s?sslmode=verify-full&sslrootcert=%s",
    var.database_username,
    urlencode(random_password.database.result),
    aws_db_instance.smartgen.address,
    aws_db_instance.smartgen.port,
    var.database_name,
    local.rds_ca_bundle_path,
  )
}

resource "aws_ecr_repository" "worker" {
  name                 = "${local.name_prefix}-worker"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.smartgen.arn
  }
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/besser/${var.environment}/smartgen-worker"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.smartgen.arn
}

resource "aws_ecs_cluster" "smartgen" {
  name = local.name_prefix

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_iam_role" "worker_execution" {
  name_prefix = "${local.name_prefix}-execution-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "worker_execution" {
  role       = aws_iam_role.worker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "worker_execution_secrets" {
  role = aws_iam_role.worker_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "kms:Decrypt",
      ]
      Resource = [
        aws_secretsmanager_secret.database_url.arn,
        aws_kms_key.smartgen.arn,
      ]
    }]
  })
}

resource "aws_iam_role" "worker_task" {
  name_prefix = "${local.name_prefix}-task-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "worker_task" {
  role = aws_iam_role.worker_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:ChangeMessageVisibility",
          "sqs:GetQueueAttributes",
        ]
        Resource = aws_sqs_queue.jobs.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
        ]
        Resource = aws_s3_bucket.artifacts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
        ]
        Resource = aws_kms_key.smartgen.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ecs:GetTaskProtection",
          "ecs:UpdateTaskProtection",
        ]
        Resource = "arn:${data.aws_partition.current.partition}:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:task/${aws_ecs_cluster.smartgen.name}/*"
      },
    ]
  })
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name_prefix}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.worker_cpu)
  memory                   = tostring(var.worker_memory)
  execution_role_arn       = aws_iam_role.worker_execution.arn
  task_role_arn            = aws_iam_role.worker_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  ephemeral_storage {
    size_in_gib = 30
  }

  container_definitions = jsonencode([{
    name                   = "smartgen-worker"
    image                  = local.worker_image
    essential              = true
    stopTimeout            = var.worker_stop_timeout_seconds
    readonlyRootFilesystem = true
    user                   = "10001:10001"
    linuxParameters = {
      initProcessEnabled = true
      capabilities = {
        drop = ["ALL"]
      }
    }
    environment = [
      { name = "BESSER_SMARTGEN_STATE_MODE", value = "production" },
      { name = "BESSER_SMARTGEN_AWS_REGION", value = var.aws_region },
      { name = "BESSER_SMARTGEN_S3_BUCKET", value = aws_s3_bucket.artifacts.id },
      { name = "BESSER_SMARTGEN_S3_PREFIX", value = "smartgen" },
      { name = "BESSER_SMARTGEN_SQS_QUEUE_URL", value = aws_sqs_queue.jobs.url },
      { name = "BESSER_SMARTGEN_KMS_KEY_ID", value = aws_kms_key.smartgen.arn },
      { name = "BESSER_SMARTGEN_S3_KMS_KEY_ID", value = aws_kms_key.smartgen.arn },
      { name = "BESSER_SMARTGEN_ALLOW_SHELL_TOOLS", value = tostring(var.worker_allow_shell_tools) },
      { name = "BESSER_SMARTGEN_APPROVAL_TIMEOUT_SECONDS", value = tostring(var.worker_approval_timeout_seconds) },
      { name = "BESSER_SMARTGEN_TASK_PROTECTION_MINUTES", value = tostring(var.worker_task_protection_minutes) },
      { name = "BESSER_SMARTGEN_TASK_PROTECTION_TIMEOUT_SECONDS", value = tostring(var.worker_task_protection_timeout_seconds) },
      { name = "BESSER_SMARTGEN_WORKER_VISIBILITY_TIMEOUT_SECONDS", value = tostring(var.worker_visibility_timeout_seconds) },
      { name = "BESSER_SMARTGEN_WORKER_MAX_ATTEMPTS", value = tostring(var.worker_max_attempts) },
      { name = "BESSER_SMARTGEN_WORKER_RETRY_DELAY_SECONDS", value = tostring(var.worker_retry_delay_seconds) },
      # readonlyRootFilesystem requires every runtime/cache/temp write to land
      # on the bounded ephemeral workspace volume prepared by the image.
      { name = "TMPDIR", value = "/workspace/tmp" },
      { name = "HOME", value = "/workspace/home" },
      { name = "XDG_CACHE_HOME", value = "/workspace/cache" },
      { name = "CARGO_HOME", value = "/workspace/cargo" },
      { name = "JAVA_TOOL_OPTIONS", value = "-Djava.io.tmpdir=/workspace/tmp" },
    ]
    secrets = [{
      name      = "BESSER_SMARTGEN_DATABASE_URL"
      valueFrom = aws_secretsmanager_secret.database_url.arn
    }]
    mountPoints = [{
      sourceVolume  = "workspace"
      containerPath = "/workspace"
      readOnly      = false
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.worker.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "worker"
      }
    }
  }])

  volume {
    name = "workspace"
  }
}

resource "aws_ecs_service" "worker" {
  name                               = "${local.name_prefix}-worker"
  cluster                            = aws_ecs_cluster.smartgen.id
  task_definition                    = aws_ecs_task_definition.worker.arn
  desired_count                      = var.worker_desired_count
  launch_type                        = "FARGATE"
  platform_version                   = "LATEST"
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  enable_execute_command             = false
  propagate_tags                     = "SERVICE"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.worker.id]
    assign_public_ip = false
  }

  lifecycle {
    ignore_changes = [desired_count]

    precondition {
      condition     = var.worker_max_count >= var.worker_desired_count
      error_message = "worker_max_count must be greater than or equal to worker_desired_count."
    }
  }
}

resource "aws_appautoscaling_target" "worker" {
  max_capacity       = var.worker_max_count
  min_capacity       = var.worker_desired_count
  resource_id        = "service/${aws_ecs_cluster.smartgen.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "worker_scale_out" {
  name               = "${local.name_prefix}-worker-scale-out"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 30
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_lower_bound = 0
      metric_interval_upper_bound = 4
      scaling_adjustment          = 1
    }

    step_adjustment {
      metric_interval_lower_bound = 4
      metric_interval_upper_bound = 9
      scaling_adjustment          = 3
    }

    step_adjustment {
      metric_interval_lower_bound = 9
      scaling_adjustment          = 5
    }
  }
}

resource "aws_appautoscaling_policy" "worker_scale_in_idle" {
  name               = "${local.name_prefix}-worker-scale-in-idle"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace

  step_scaling_policy_configuration {
    adjustment_type         = "ExactCapacity"
    cooldown                = 300
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = var.worker_desired_count
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "worker_scale_out" {
  alarm_name          = "${local.name_prefix}-worker-scale-out"
  alarm_description   = "Visible SmartGen backlog requires additional isolated workers"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_appautoscaling_policy.worker_scale_out.arn]

  dimensions = {
    QueueName = aws_sqs_queue.jobs.name
  }
}

# Scaling on the visible queue alone can terminate a worker while its message
# is hidden. Scale in only after both queued and in-flight work reach zero.
resource "aws_cloudwatch_metric_alarm" "worker_scale_in_idle" {
  alarm_name          = "${local.name_prefix}-worker-scale-in-idle"
  alarm_description   = "SmartGen queue and in-flight workload are both idle"
  evaluation_periods  = 5
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_appautoscaling_policy.worker_scale_in_idle.arn]

  metric_query {
    id          = "outstanding"
    expression  = "visible + inflight"
    label       = "Outstanding SmartGen jobs"
    return_data = true
  }

  metric_query {
    id          = "visible"
    return_data = false
    metric {
      namespace   = "AWS/SQS"
      metric_name = "ApproximateNumberOfMessagesVisible"
      period      = 60
      stat        = "Maximum"
      dimensions = {
        QueueName = aws_sqs_queue.jobs.name
      }
    }
  }

  metric_query {
    id          = "inflight"
    return_data = false
    metric {
      namespace   = "AWS/SQS"
      metric_name = "ApproximateNumberOfMessagesNotVisible"
      period      = 60
      stat        = "Maximum"
      dimensions = {
        QueueName = aws_sqs_queue.jobs.name
      }
    }
  }
}

resource "aws_iam_policy" "controller" {
  name_prefix = "${local.name_prefix}-controller-"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:GetQueueAttributes",
        ]
        Resource = aws_sqs_queue.jobs.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
        ]
        Resource = aws_s3_bucket.artifacts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
        ]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey",
        ]
        Resource = aws_kms_key.smartgen.arn
      },
      {
        Effect   = "Allow"
        Action   = "secretsmanager:GetSecretValue"
        Resource = aws_secretsmanager_secret.database_url.arn
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "controller" {
  count      = var.controller_instance_role_name == "" ? 0 : 1
  role       = var.controller_instance_role_name
  policy_arn = aws_iam_policy.controller.arn
}

resource "aws_cloudwatch_metric_alarm" "queue_age" {
  alarm_name          = "${local.name_prefix}-queue-age"
  alarm_description   = "SmartGen jobs have waited longer than five minutes"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateAgeOfOldestMessage"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 5
  threshold           = 300
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions

  dimensions = {
    QueueName = aws_sqs_queue.jobs.name
  }
}

resource "aws_cloudwatch_metric_alarm" "dead_letter" {
  alarm_name          = "${local.name_prefix}-dead-letter"
  alarm_description   = "At least one SmartGen job reached the dead-letter queue"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions

  dimensions = {
    QueueName = aws_sqs_queue.dead_letter.name
  }
}
