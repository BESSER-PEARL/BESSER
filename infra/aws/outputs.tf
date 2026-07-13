output "smartgen_database_url_secret_arn" {
  value = aws_secretsmanager_secret.database_url.arn
}

output "smartgen_artifact_bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "smartgen_queue_url" {
  value = aws_sqs_queue.jobs.url
}

output "smartgen_dead_letter_queue_url" {
  value = aws_sqs_queue.dead_letter.url
}

output "smartgen_kms_key_arn" {
  value = aws_kms_key.smartgen.arn
}

output "worker_ecr_repository" {
  value = aws_ecr_repository.worker.repository_url
}

output "worker_ecs_cluster_arn" {
  value = aws_ecs_cluster.smartgen.arn
}

output "worker_task_definition_arn" {
  value = aws_ecs_task_definition.worker.arn
}

output "worker_ecs_service_name" {
  value = aws_ecs_service.worker.name
}

output "worker_security_group_id" {
  value = aws_security_group.worker.id
}

output "controller_policy_arn" {
  value = aws_iam_policy.controller.arn
}
