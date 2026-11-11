"""Reading a repo file over 1 MB, such as a large ``buml/diagrams.json``."""

import asyncio
import base64
import json

import httpx

from besser.utilities.web_modeling_editor.backend.services.deployment import github_service
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService


def _serve(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(github_service.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler)))


def test_a_file_over_one_megabyte_is_read_through_the_raw_media_type(monkeypatch):
    """GitHub's JSON contents response leaves ``content`` empty over 1 MB, so a big model read back as None."""
    big = json.dumps({"project": {"name": "Big", "notes": "x" * 1_200_000}})

    def github(request: httpx.Request) -> httpx.Response:
        if request.headers["accept"] == "application/vnd.github.raw+json":
            return httpx.Response(200, content=big.encode("utf-8"))
        return httpx.Response(200, json={"sha": "s1", "path": "buml/diagrams.json", "size": len(big),
                                         "encoding": "none", "content": ""})

    _serve(monkeypatch, github)
    result = asyncio.run(GitHubService("token").get_file_content("me", "app", "buml/diagrams.json", ref="abc"))

    assert result["content"] == big and result["sha"] == "s1"


def test_a_small_file_is_still_read_from_the_json_response(monkeypatch):
    calls = []

    def github(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["accept"])
        return httpx.Response(200, json={"sha": "s2", "path": "a.txt", "size": 2, "encoding": "base64",
                                         "content": base64.b64encode(b"hi").decode()})

    _serve(monkeypatch, github)
    result = asyncio.run(GitHubService("token").get_file_content("me", "app", "a.txt"))

    assert result["content"] == "hi" and calls == ["application/vnd.github+json"]
