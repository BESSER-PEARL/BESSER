"""``besser mcp``: the commands as MCP tools, spoken to over stdio as a client would."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from besser.cli.mcp_server import _option_assignments
from tests.cli.conftest import CLASS_DIAGRAM, LIBRARY_MODEL

_TIMEOUT = 120


class McpServer:
    """A ``besser mcp`` process and a minimal JSON-RPC client for it, one request at a time."""

    def __init__(self, root, *flags):
        # A file, not a pipe: nobody reads standard error while the server runs, and a full pipe would block it.
        self.stderr = tempfile.TemporaryFile(mode="w+")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "besser", "mcp", "--root", str(root), *flags],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True,
        )
        self.next_id = 0
        self.initialize = self.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"},
        })
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method, params=None):
        """Send a request and return its response; every line on standard output must be JSON-RPC."""
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, **({"params": params} if params else {})})
        while True:
            line = self.process.stdout.readline()
            assert line, f"the server exited: {self.read_stderr()}"
            message = json.loads(line)
            assert message["jsonrpc"] == "2.0"
            if message.get("id") == self.next_id:
                return message

    def call(self, tool, **arguments):
        """The result of ``tools/call``, with its JSON text checked against its structured content."""
        result = self.request("tools/call", {"name": tool, "arguments": arguments})["result"]
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
        assert result["isError"] is (not result["structuredContent"]["ok"])
        return result["structuredContent"]

    def read_stderr(self):
        self.stderr.seek(0)
        return self.stderr.read()

    def close(self):
        """Stop the server (end of input) and return what it wrote on standard error."""
        if self.stderr.closed:
            return ""
        self.process.stdin.close()
        self.process.wait(timeout=_TIMEOUT)
        self.process.stdout.close()
        stderr = self.read_stderr()
        self.stderr.close()
        return stderr


@pytest.fixture
def serve(tmp_path):
    """Start ``besser mcp --root tmp_path`` with the given flags; stopped at the end of the test."""
    pytest.importorskip("mcp")
    servers = []

    def start(*flags, root=tmp_path):
        servers.append(McpServer(root, *flags))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


@pytest.fixture
def json_model(write):
    return write("library.json", CLASS_DIAGRAM)


def test_it_lists_the_commands_as_tools(serve):
    server = serve()
    tools = {tool["name"]: tool for tool in server.request("tools/list")["result"]["tools"]}

    assert server.initialize["result"]["serverInfo"]["name"] == "besser"
    assert {name: tool["inputSchema"]["required"] for name, tool in tools.items()} == {
        "validate": ["paths"], "info": ["model"], "generators": [],
        "generate": ["generator", "model", "output_dir"], "regenerate": ["generator", "model", "output_dir"],
        "convert": ["input", "output"],
    }
    assert "pydantic" in tools["generate"]["inputSchema"]["properties"]["generator"]["enum"]
    assert tools["validate"]["annotations"]["readOnlyHint"] is True
    assert tools["generate"]["annotations"]["destructiveHint"] is True


def test_validate_and_generate_return_the_cli_json(serve, json_model, tmp_path):
    server = serve()

    validated = server.call("validate", paths=[json_model])
    generated = server.call("generate", generator="pydantic", model=json_model, output_dir="out")
    sqlalchemy = server.call("generate", generator="sqlalchemy", model=json_model, output_dir="db",
                             options={"dbms": "postgresql"})

    assert validated["ok"] is True and validated["command"] == "validate"
    assert [model["kind"] for model in validated["models"]] == ["domain"]
    assert generated["ok"] is True and generated["errors"] == []
    assert generated["files"] == [(tmp_path / "out" / "pydantic_classes.py").as_posix()]
    assert "class Book" in (tmp_path / "out" / "pydantic_classes.py").read_text()
    assert sqlalchemy["ok"] is True
    assert "postgresql" in (tmp_path / "db" / "sql_alchemy.py").read_text()


def test_generate_refuses_a_folder_that_is_not_empty_unless_forced(serve, json_model, write, tmp_path):
    write("out/keep.txt", "mine")
    server = serve()

    refused = server.call("generate", generator="pydantic", model=json_model, output_dir="out")
    forced = server.call("generate", generator="pydantic", model=json_model, output_dir="out", force=True)

    assert refused["ok"] is False and refused["exit_code"] == 2
    assert "is not empty" in refused["errors"][0]
    assert forced["ok"] is True


def test_paths_outside_the_root_are_refused(serve, json_model, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.json").symlink_to(tmp_path / json_model)
    (root / "out").mkdir()
    (root / "out" / "escape").symlink_to(tmp_path)
    server = serve(root=root)

    refusals = [
        server.call("info", model=f"../{json_model}"),
        server.call("info", model=str(tmp_path / json_model)),
        server.call("info", model="link.json"),
        server.call("validate", paths=["sub/../../library.json"]),
        server.call("generate", generator="pydantic", model="../library.json", output_dir="out"),
        server.call("convert", input="../library.json", output="model.py"),
    ]
    into_link = server.call("generate", generator="pydantic", model="x.json", output_dir="out", force=True)
    output_outside = server.call("convert", input="x.json", output="../model.py")

    for result in refusals + [output_outside]:
        assert result["ok"] is False and result["exit_code"] == 2
        assert "outside the server's root" in result["errors"][0]
    assert "symbolic link to a path outside" in into_link["errors"][0]
    assert not (tmp_path / "model.py").exists()


def test_standard_input_is_not_a_model_file_over_mcp(serve):
    result = serve().call("info", model="-")

    assert result["exit_code"] == 2 and "standard input" in result["errors"][0]


def test_py_models_need_exec_py(serve, write, tmp_path):
    # It prints while loading: that must reach standard error, not the protocol on standard output.
    model = write("library.py", "print('loading the model')\n" + LIBRARY_MODEL)

    refused = serve().call("info", model=model)
    server = serve("--exec-py")
    accepted = server.call("info", model=model)

    assert refused["ok"] is False and refused["exit_code"] == 2
    assert "--exec-py" in refused["errors"][0]
    assert accepted["ok"] is True
    assert accepted["models"][0]["name"] == "library_model"
    assert "loading the model" in server.close()


def test_invalid_arguments_are_a_tool_error(serve, json_model):
    result = serve().call("generate", generator="nope", model=json_model, output_dir="out")

    assert result["exit_code"] == 2 and result["errors"][0].startswith("invalid arguments: generator:")


def test_without_the_sdk_it_says_what_to_install(tmp_path):
    shim = tmp_path / "shim"
    (shim / "mcp").mkdir(parents=True)
    (shim / "mcp" / "__init__.py").write_text("raise ModuleNotFoundError(\"No module named 'mcp'\", name='mcp')\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(shim), os.environ.get("PYTHONPATH")]))}

    run = subprocess.run([sys.executable, "-m", "besser", "mcp", "--root", str(tmp_path)], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=_TIMEOUT, check=False)

    assert run.returncode == 2
    assert 'pip install "besser[mcp]"' in run.stderr


def test_options_become_cli_assignments():
    assert _option_assignments({"dbms": "postgresql", "nested": True, "http_methods": ["GET", "POST"], "n": 2}) == [
        "dbms=postgresql", "nested=true", "http_methods=GET,POST", "n=2",
    ]
    assert _option_assignments(["dbms=sqlite"]) == ["dbms=sqlite"]
    assert _option_assignments(None) == []
