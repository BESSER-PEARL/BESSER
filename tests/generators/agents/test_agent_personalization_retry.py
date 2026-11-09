"""``call_openai_chat`` retries with besser.llm's one policy, not the SDK's.

The SDK's default retries sent a spent quota (429 ``insufficient_quota``) three
times; the shared policy fails it at once and still retries a transient error.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

openai = pytest.importorskip("openai")

from besser.generators.agents import agent_personalization  # noqa: E402

QUOTA = (429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota",
                         "code": "insufficient_quota"}})
UNAVAILABLE = (503, {"error": {"message": "The server is overloaded", "type": "server_error"}})
OK = (200, {"id": "c", "object": "chat.completion", "created": 0, "model": "gpt-5",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": " Bonjour "}}]})


@pytest.fixture
def openai_server(monkeypatch):
    """A local chat-completions endpoint answering with the queued responses."""
    replies, hits = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            hits.append(self.path)
            status, body = replies.pop(0) if len(replies) > 1 else replies[0]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
    monkeypatch.setattr(agent_personalization.time, "sleep", lambda _seconds: None)
    yield replies, hits
    server.shutdown()
    server.server_close()


def test_a_spent_quota_is_sent_once(openai_server):
    replies, hits = openai_server
    replies.append(QUOTA)

    # The SDK's message format varies by version; the status is stable.
    with pytest.raises(openai.RateLimitError) as raised:
        agent_personalization.call_openai_chat("sys", "user", openai_api_key="sk-test")
    assert raised.value.status_code == 429

    assert len(hits) == 1


def test_a_transient_error_is_retried_until_it_succeeds(openai_server, caplog):
    replies, hits = openai_server
    replies.extend([UNAVAILABLE, UNAVAILABLE, OK])

    assert agent_personalization.call_openai_chat("sys", "user", openai_api_key="sk-test") == "Bonjour"
    assert len(hits) == 3
    assert [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()] == [
        "OpenAI call failed (attempt 1/3), retrying in 0.5s: InternalServerError",
        "OpenAI call failed (attempt 2/3), retrying in 1.0s: InternalServerError",
    ]


def test_a_transient_error_stops_after_the_last_retry(openai_server):
    replies, hits = openai_server
    replies.append(UNAVAILABLE)

    with pytest.raises(Exception, match="overloaded"):
        agent_personalization.call_openai_chat("sys", "user", openai_api_key="sk-test")

    assert len(hits) == 3
