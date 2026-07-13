variable "aws_region" {
  description = "AWS region containing the existing BESSER deployment."
  type        = string
  default     = "eu-north-1"
}

variable "environment" {
  description = "Environment name used in resource names and tags."
  type        = string
  default     = "experimental"
}

variable "vpc_id" {
  description = "Existing VPC used by the EC2 controller and Fargate workers."
  type        = string
}

variable "private_subnet_ids" {
  description = "At least two private subnets in distinct availability zones."
  type        = list(string)

  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "private_subnet_ids must contain at least two subnets."
  }
}

variable "controller_security_group_id" {
  description = "Security group attached to the existing BESSER EC2 controller."
  type        = string
}

variable "controller_instance_role_name" {
  description = "Optional existing EC2 IAM role to receive the SmartGen controller policy."
  type        = string
  default     = ""
}

variable "worker_image" {
  description = "Required immutable ECR worker image pinned by sha256 digest."
  type        = string

  validation {
    condition = can(regex(
      "^[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$",
      var.worker_image,
    ))
    error_message = "worker_image must be an ECR image pinned by sha256 digest."
  }
}

variable "database_name" {
  type    = string
  default = "besser_smartgen"
}

variable "database_username" {
  type    = string
  default = "besser_smartgen"
}

variable "database_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "database_multi_az" {
  type    = bool
  default = false
}

variable "database_deletion_protection" {
  type    = bool
  default = true
}

variable "artifact_retention_days" {
  type    = number
  default = 7
}

variable "worker_cpu" {
  description = "Fargate task CPU units."
  type        = number
  default     = 2048
}

variable "worker_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 4096
}

variable "worker_desired_count" {
  description = "Minimum number of supervised SmartGen worker tasks."
  type        = number
  default     = 1

  validation {
    condition     = floor(var.worker_desired_count) == var.worker_desired_count && var.worker_desired_count >= 1 && var.worker_desired_count <= 10
    error_message = "worker_desired_count must be a whole number between 1 and 10."
  }
}

variable "worker_max_count" {
  description = "Maximum worker count allowed by queue-backlog autoscaling."
  type        = number
  default     = 10

  validation {
    condition     = floor(var.worker_max_count) == var.worker_max_count && var.worker_max_count >= 1 && var.worker_max_count <= 100
    error_message = "worker_max_count must be a whole number between 1 and 100."
  }
}

variable "worker_allow_shell_tools" {
  description = "Enable approval-gated arbitrary shell tools in the shared-IAM worker task. Single-tenant only."
  type        = bool
  default     = false
}

variable "worker_approval_timeout_seconds" {
  description = "Maximum time an isolated worker waits for one shell-tool approval."
  type        = number
  default     = 600

  validation {
    condition     = floor(var.worker_approval_timeout_seconds) == var.worker_approval_timeout_seconds && var.worker_approval_timeout_seconds >= 30 && var.worker_approval_timeout_seconds <= 3600
    error_message = "worker_approval_timeout_seconds must be a whole number between 30 and 3600."
  }
}

variable "worker_task_protection_minutes" {
  description = "ECS scale-in/deployment protection duration refreshed while paid work is active."
  type        = number
  default     = 30

  validation {
    condition     = floor(var.worker_task_protection_minutes) == var.worker_task_protection_minutes && var.worker_task_protection_minutes >= 1 && var.worker_task_protection_minutes <= 2880
    error_message = "worker_task_protection_minutes must be a whole number between 1 and 2880."
  }
}

variable "worker_task_protection_timeout_seconds" {
  description = "Bounded timeout for one ECS agent task-protection request."
  type        = number
  default     = 3

  validation {
    condition     = var.worker_task_protection_timeout_seconds >= 1 && var.worker_task_protection_timeout_seconds <= 10
    error_message = "worker_task_protection_timeout_seconds must be between 1 and 10."
  }
}

variable "worker_stop_timeout_seconds" {
  description = "Grace period for checkpoint and cleanup after ECS sends SIGTERM."
  type        = number
  default     = 120

  validation {
    condition     = floor(var.worker_stop_timeout_seconds) == var.worker_stop_timeout_seconds && var.worker_stop_timeout_seconds >= 2 && var.worker_stop_timeout_seconds <= 120
    error_message = "worker_stop_timeout_seconds must be a whole number between 2 and 120."
  }
}

variable "worker_visibility_timeout_seconds" {
  description = "SQS visibility window renewed by a running worker."
  type        = number
  default     = 3600

  validation {
    condition     = floor(var.worker_visibility_timeout_seconds) == var.worker_visibility_timeout_seconds && var.worker_visibility_timeout_seconds >= 60 && var.worker_visibility_timeout_seconds <= 43200
    error_message = "worker_visibility_timeout_seconds must be a whole number between 60 and 43200."
  }
}

variable "worker_max_attempts" {
  description = "Worker retry limit and SQS redrive threshold."
  type        = number
  default     = 3

  validation {
    condition     = floor(var.worker_max_attempts) == var.worker_max_attempts && var.worker_max_attempts >= 1 && var.worker_max_attempts <= 10
    error_message = "worker_max_attempts must be a whole number between 1 and 10."
  }
}

variable "worker_retry_delay_seconds" {
  description = "Delay before a retryable job becomes visible again."
  type        = number
  default     = 5

  validation {
    condition     = floor(var.worker_retry_delay_seconds) == var.worker_retry_delay_seconds && var.worker_retry_delay_seconds >= 0 && var.worker_retry_delay_seconds <= 900
    error_message = "worker_retry_delay_seconds must be a whole number between 0 and 900."
  }
}

variable "alarm_sns_topic_arn" {
  description = "Optional SNS topic receiving queue and database alarms."
  type        = string
  default     = ""
}
