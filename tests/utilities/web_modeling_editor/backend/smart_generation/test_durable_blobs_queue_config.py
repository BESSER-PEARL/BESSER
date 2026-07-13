import asyncio
import io
import pytest

from besser.utilities.web_modeling_editor.backend.services.smart_generation.durable_state import (
    ArtifactRef,
    DurableStateConfig,
    DurableStateConfigurationError,
    DurableStateMode,
    FileSystemBlobStore,
    InMemoryJobQueue,
    JobMessage,
    MissingOptionalDependencyError,
    S3BlobStore,
    SQSJobQueue,
    StorageIntegrityError,
    build_durable_state,
)


class FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.upload_args = None

    def head_bucket(self, **kwargs):
        return kwargs

    def upload_file(self, source, bucket, key, ExtraArgs):
        self.objects[(bucket, key)] = open(source, "rb").read()
        self.upload_args = ExtraArgs

    def download_file(self, bucket, key, destination):
        with open(destination, "wb") as handle:
            handle.write(self.objects[(bucket, key)])

    def put_object(self, **kwargs):
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = bytes(kwargs["Body"])
        return {}

    def get_object(self, **kwargs):
        data = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        import hashlib

        return {"Body": io.BytesIO(data), "Metadata": {"sha256": hashlib.sha256(data).hexdigest()}}

    def delete_object(self, **kwargs):
        self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://signed.example/{Params['Key']}?ttl={ExpiresIn}"


class FakeSQSClient:
    def __init__(self):
        self.sent = []
        self.deleted = []
        self.released = []

    def get_queue_attributes(self, **kwargs):
        return {"Attributes": {"SqsManagedSseEnabled": "true"}}

    def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return {"MessageId": "message-1"}

    def receive_message(self, **kwargs):
        if not self.sent:
            return {}
        request = self.sent.pop(0)
        return {
            "Messages": [{
                "Body": request["MessageBody"],
                "ReceiptHandle": "receipt-1",
                "MessageId": "message-1",
                "Attributes": {"ApproximateReceiveCount": "2"},
            }]
        }

    def delete_message(self, **kwargs):
        self.deleted.append(kwargs)

    def change_message_visibility(self, **kwargs):
        self.released.append(kwargs)


class UnencryptedSQSClient(FakeSQSClient):
    def get_queue_attributes(self, **kwargs):
        return {"Attributes": {}}


def test_filesystem_artifact_and_checkpoint_storage(tmp_path):
    async def scenario():
        store = FileSystemBlobStore(str(tmp_path / "blobs"))
        await store.initialize()
        source = tmp_path / "app.zip"
        source.write_bytes(b"generated-app")
        artifact = await store.put_artifact(
            "github:1234",
            "a" * 32,
            str(source),
            file_name='unsafe\r\n"name.zip',
            content_type="application/zip",
        )
        assert artifact.file_name == "unsafe_name.zip"
        destination = tmp_path / "download.zip"
        await store.download_artifact(artifact, str(destination))
        assert destination.read_bytes() == b"generated-app"
        assert await store.create_download_url(artifact, expires_seconds=30) is None

        checkpoint = await store.put_checkpoint("github:1234", "a" * 32, b"checkpoint")
        assert checkpoint.size_bytes == 10
        assert await store.get_checkpoint("github:1234", "a" * 32) == b"checkpoint"
        checkpoint_path = tmp_path / "blobs" / checkpoint.storage_key
        checkpoint_path.write_bytes(b"tampered")
        with pytest.raises(StorageIntegrityError, match="Checkpoint checksum"):
            await store.get_checkpoint("github:1234", "a" * 32)
        await store.put_checkpoint("github:1234", "a" * 32, b"checkpoint")
        await store.delete_checkpoint("github:1234", "a" * 32)
        assert await store.get_checkpoint("github:1234", "a" * 32) is None

        with pytest.raises(StorageIntegrityError):
            await store.download_artifact(
                ArtifactRef("../escape", "x", 1, "0" * 64, "text/plain", 0),
                str(tmp_path / "escape"),
            )

    asyncio.run(scenario())


def test_s3_adapter_encrypts_objects_and_presigns(tmp_path):
    async def scenario():
        client = FakeS3Client()
        store = S3BlobStore(
            "smartgen-artifacts-test",
            kms_key_id="kms-key",
            client=client,
        )
        await store.initialize()
        source = tmp_path / "app.zip"
        source.write_bytes(b"app")
        artifact = await store.put_artifact(
            "github:1234",
            "a" * 32,
            str(source),
            file_name="app.zip",
            content_type="application/zip",
        )
        assert client.upload_args["ServerSideEncryption"] == "aws:kms"
        assert client.upload_args["SSEKMSKeyId"] == "kms-key"
        assert "github:1234" not in artifact.storage_key
        assert await store.create_download_url(artifact, expires_seconds=300)
        await store.put_checkpoint("github:1234", "a" * 32, b"checkpoint")
        assert await store.get_checkpoint("github:1234", "a" * 32) == b"checkpoint"

    asyncio.run(scenario())


def test_job_queues_reject_plaintext_secrets_and_support_ack_release():
    async def scenario():
        queue = InMemoryJobQueue()
        message = JobMessage("a" * 32, "github:1234", {"provider": "openai"})
        first_id = await queue.enqueue(message, deduplication_id="request-1")
        assert await queue.enqueue(message, deduplication_id="request-1") == first_id
        received = (await queue.receive(wait_seconds=0))[0]
        assert await queue.extend_visibility(received, visibility_timeout=60) is True
        await queue.release(received)
        assert await queue.extend_visibility(received, visibility_timeout=60) is False
        retried = (await queue.receive(wait_seconds=0))[0]
        assert retried.receive_count == 2
        await queue.acknowledge(retried)
        assert await queue.receive(wait_seconds=0) == ()
        with pytest.raises(ValueError, match="plaintext secrets"):
            await queue.enqueue(JobMessage("b" * 32, "github:1234", {"api_key": "sk-live"}))

        fake = FakeSQSClient()
        sqs = SQSJobQueue(
            "https://sqs.eu-north-1.amazonaws.com/123/smartgen.fifo",
            client=fake,
        )
        await sqs.initialize()
        await sqs.enqueue(message, deduplication_id="request-1")
        assert fake.sent[0]["MessageDeduplicationId"] == "request-1"
        sqs_job = (await sqs.receive())[0]
        assert sqs_job.message.run_id == message.run_id
        assert await sqs.extend_visibility(sqs_job, visibility_timeout=120) is True
        await sqs.release(sqs_job, delay_seconds=5)
        await sqs.acknowledge(sqs_job)
        assert fake.released and fake.deleted

        unencrypted = SQSJobQueue(
            "https://sqs.eu-north-1.amazonaws.com/123/unencrypted",
            client=UnencryptedSQSClient(),
        )
        with pytest.raises(DurableStateConfigurationError, match="must enable"):
            await unencrypted.initialize()

    asyncio.run(scenario())


def test_config_is_strict_and_production_does_not_fallback(monkeypatch, tmp_path):
    local = DurableStateConfig.from_env({
        "BESSER_SMARTGEN_STATE_MODE": "local",
        "BESSER_SMARTGEN_SQLITE_PATH": str(tmp_path / "state.sqlite3"),
        "BESSER_SMARTGEN_STORAGE_DIR": str(tmp_path / "blobs"),
    })
    assert local.mode == DurableStateMode.LOCAL
    assert local.max_request_bytes == 5 * 1024 * 1024
    assert build_durable_state(local).config is local

    with pytest.raises(DurableStateConfigurationError, match="DATABASE_URL"):
        DurableStateConfig.from_env({"BESSER_SMARTGEN_STATE_MODE": "production"})

    with pytest.raises(DurableStateConfigurationError, match="verify-full"):
        DurableStateConfig.from_env({
            "BESSER_SMARTGEN_STATE_MODE": "production",
            "BESSER_SMARTGEN_DATABASE_URL": "postgresql://db/smartgen",
            "BESSER_SMARTGEN_S3_BUCKET": "smartgen-artifacts-prod",
            "BESSER_SMARTGEN_SQS_QUEUE_URL": "https://sqs.example.com/123/smartgen",
        })

    with pytest.raises(DurableStateConfigurationError, match="sslrootcert"):
        DurableStateConfig.from_env({
            "BESSER_SMARTGEN_STATE_MODE": "production",
            "BESSER_SMARTGEN_DATABASE_URL": "postgresql://db/smartgen?sslmode=verify-full",
            "BESSER_SMARTGEN_S3_BUCKET": "smartgen-artifacts-prod",
            "BESSER_SMARTGEN_SQS_QUEUE_URL": "https://sqs.example.com/123/smartgen",
        })

    production = DurableStateConfig.from_env({
        "BESSER_SMARTGEN_STATE_MODE": "production",
        "BESSER_SMARTGEN_DATABASE_URL": (
            "postgresql://db/smartgen?sslmode=verify-full"
            "&sslrootcert=/opt/aws/rds/global-bundle.pem"
        ),
        "BESSER_SMARTGEN_S3_BUCKET": "smartgen-artifacts-prod",
        "BESSER_SMARTGEN_SQS_QUEUE_URL": "https://sqs.example.com/123/smartgen",
    })

    def missing_dependency(*_args, **_kwargs):
        raise MissingOptionalDependencyError("psycopg missing")

    monkeypatch.setattr(
        "besser.utilities.web_modeling_editor.backend.services.smart_generation."
        "durable_state.postgres_store.require_optional_dependency",
        missing_dependency,
    )
    with pytest.raises(MissingOptionalDependencyError, match="psycopg"):
        build_durable_state(production)
