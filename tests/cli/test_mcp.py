"""``besser mcp``: the commands as MCP tools, spoken to over stdio as a client would."""

import importlib.util
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from besser.cli.commands import option_assignments as _option_assignments
from besser.cli.mcp_server import for_client
from tests.cli.conftest import CLASS_DIAGRAM, LIBRARY_MODEL

_TIMEOUT = 120
# How long a response may take: a server that hangs fails the test instead of blocking it.
_RESPONSE_TIMEOUT = 60

needs_pygit2 = pytest.mark.skipif(importlib.util.find_spec("pygit2") is None,
                                  reason='regenerate merges with pygit2: pip install "besser[regen]"')


class McpServer:
    """A ``besser mcp`` process and a minimal JSON-RPC client for it, one request at a time."""

    def __init__(self, root, *flags):
        # A file, not a pipe: nobody reads standard error while the server runs, and a full pipe would block it.
        self.stderr = tempfile.TemporaryFile(mode="w+")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "besser", "mcp", "--root", str(root), *flags],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True,
        )
        # Lines of standard output, read by a thread so that a response can be waited for with a timeout.
        self.lines = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self.next_id = 0
        self.initialize = self.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"},
        })
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _read_stdout(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put("")  # end of output

    def readline(self):
        try:
            return self.lines.get(timeout=_RESPONSE_TIMEOUT)
        except queue.Empty:
            pytest.fail(f"no response in {_RESPONSE_TIMEOUT} s: {self.read_stderr()}")

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method, params=None):
        """Send a request and return its response; every line on standard output must be JSON-RPC."""
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, **({"params": params} if params else {})})
        while True:
            line = self.readline()
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
        """Stop the server (end of input) and return what it wrote on standard error.

        Nothing may follow the session on standard output: it is the protocol's channel.
        """
        if self.stderr.closed:
            return ""
        self.process.stdin.close()
        self.process.wait(timeout=_TIMEOUT)
        after = []
        while (line := self.readline()) != "":
            after.append(line)
        self.process.stdout.close()
        stderr = self.read_stderr()
        self.stderr.close()
        assert after == [], f"printed on standard output after the session: {after}"
        assert self.process.returncode == 0, stderr
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
        "new": ["path"], "schema": [], "validate": ["paths"], "info": ["model"], "generators": [],
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
    # Paths as the client gave them: relative to the root.
    assert generated["files"] == ["out/pydantic_classes.py"]
    assert generated["model"] == "library.json" and generated["output_dir"] == "out"
    assert validated["model_files"] == ["library.json"]
    assert "class Book" in (tmp_path / "out" / "pydantic_classes.py").read_text()
    assert sqlalchemy["ok"] is True
    assert "postgresql" in (tmp_path / "db" / "sql_alchemy.py").read_text()


def test_generate_refuses_a_folder_that_is_not_empty_unless_forced(serve, json_model, write, tmp_path):
    write("out/keep.txt", "mine")
    server = serve()

    refused = server.call("generate", generator="pydantic", model=json_model, output_dir="out")
    forced = server.call("generate", generator="pydantic", model=json_model, output_dir="out", force=True)

    assert refused["ok"] is False and refused["exit_code"] == 2
    [message] = refused["errors"]
    assert message.startswith("out is not empty and generating overwrites what is there: pass `force: true`")
    assert "the `regenerate` tool" in message and "--" not in message and "besser" not in message
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


def test_messages_name_the_tools_and_their_arguments_and_paths_are_root_relative(serve, json_model, tmp_path):
    server = serve()
    server.call("generate", generator="pydantic", model=json_model, output_dir="out")
    again = server.call("generate", generator="pydantic", model=json_model, output_dir="out")

    [message] = again["errors"]
    assert message.startswith("out holds generated code and the record of it (.besser_generated.json): "
                              'the `regenerate` tool with {"generator": "pydantic", "model": "library.json", '
                              '"output_dir": "out"} brings it up')
    assert message.endswith("`force: true` overwrites them") and str(tmp_path) not in message


@needs_pygit2
def test_regenerate_keeps_an_edit_over_mcp(serve, json_model, tmp_path):
    server = serve()
    server.call("generate", generator="pydantic", model=json_model, output_dir="out")
    generated = tmp_path / "out" / "pydantic_classes.py"
    generated.write_text(generated.read_text() + "\n# my edit\n")

    planned = server.call("regenerate", generator="pydantic", model=json_model, output_dir="out", dry_run=True)
    regenerated = server.call("regenerate", generator="pydantic", model=json_model, output_dir="out")

    assert planned["ok"] is True and planned["dry_run"] is True and planned["files"] == []
    assert regenerated["ok"] is True and regenerated["command"] == "regenerate"
    assert regenerated["output_dir"] == "out"
    assert "# my edit" in generated.read_text()


def test_regenerate_lists_only_the_generators_that_support_it(serve):
    tools = {tool["name"]: tool for tool in serve().request("tools/list")["result"]["tools"]}

    regenerable = tools["regenerate"]["inputSchema"]["properties"]["generator"]["enum"]
    assert "pydantic" in regenerable
    assert set(regenerable) < set(tools["generate"]["inputSchema"]["properties"]["generator"]["enum"])


def test_generate_dry_run_writes_nothing_and_reports_root_relative_paths(serve, json_model, tmp_path):
    result = serve().call("generate", generator="pydantic", model=json_model, output_dir="out", dry_run=True)

    assert result["ok"] is True and result["files"] == []
    assert result["would_write"]["out/pydantic_classes.py"] == "new"
    assert not (tmp_path / "out").exists()


def test_convert_writes_the_output_under_the_root(serve, json_model, tmp_path):
    result = serve().call("convert", input=json_model, output="model.py")

    assert result["ok"] is True, result["errors"]
    assert result["files"] == ["model.py"]
    assert result["input"] == "library.json" and result["output"] == "model.py"
    assert "DomainModel(" in (tmp_path / "model.py").read_text()


def test_an_unknown_tool_is_a_protocol_error(serve):
    response = serve().request("tools/call", {"name": "frobnicate", "arguments": {}})

    assert "unknown tool 'frobnicate'" in json.dumps(response)
    assert "result" not in response or response["result"].get("isError") is True


def test_a_symbolic_link_loop_in_a_folder_is_skipped_as_the_command_does(serve, json_model, tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "library.json").write_text((tmp_path / json_model).read_text())
    (tmp_path / "models" / "a").symlink_to(tmp_path / "models" / "b")
    (tmp_path / "models" / "b").symlink_to(tmp_path / "models" / "a")

    result = serve().call("validate", paths=["models"])

    assert result["ok"] is True, result
    assert result["model_files"] == ["models/library.json"]


def test_nothing_follows_the_session_on_standard_output_with_json(serve):
    server = serve("--json")
    server.request("tools/list")
    server.close()  # asserts that standard output ends with the session


@pytest.mark.parametrize("flags", [[], ["--json"], ["-q"], ["-v"]])
def test_a_root_that_does_not_exist_is_a_usage_error(tmp_path, flags):
    run = subprocess.run([sys.executable, "-m", "besser", "mcp", "--root", "missing", *flags], cwd=tmp_path,
                         capture_output=True, text=True, timeout=_TIMEOUT, check=False, stdin=subprocess.DEVNULL)

    assert run.returncode == 2
    if "--json" in flags:
        assert json.loads(run.stdout)["command"] == "mcp"
        assert json.loads(run.stdout)["errors"] == ["--root missing: no such directory"]
    else:
        assert run.stdout == "" and run.stderr == "error: --root missing: no such directory\n"


def test_for_client_makes_paths_relative_and_names_the_tool_arguments():
    root = Path("/work/project")
    output = {
        "files": ["/work/project/out/a.py"], "output_dir": "/work/project/out", "model": "/work/project",
        "would_write": {"/work/project/out/a.py": "new"},
        "errors": ["/work/project/m.json holds two: name it with --use NAME; --option takes name=value"],
        "warnings": ["/work/projects/x stays: 'besser regenerate sql /work/project/m.json -o out --adopt'"],
        "notes": ["2 .py files skipped; pass --exec-py to load them", "'besser completion' is not a tool",
                  "\"besser regenerate rest_api m.json -o out --option 'http_methods=GET, POST'\" keeps edits"],
        "results": [{"errors": ["pass --force to overwrite, --dry-run and --strict"]}],
        "generators": [{"description": "keeps --force as it is"}],
    }

    assert for_client(output, root) == {
        "files": ["out/a.py"], "output_dir": "out", "model": ".",
        "would_write": {"out/a.py": "new"},
        "errors": ["m.json holds two: name it with the `use` argument; the `options` argument takes name=value"],
        "warnings": ['/work/projects/x stays: the `regenerate` tool with {"generator": "sql", "model": "m.json", '
                     '"output_dir": "out", "adopt": true}'],
        "notes": ["2 .py files skipped; restart the server with --exec-py to load them", "'besser completion' is not a tool",
                  'the `regenerate` tool with {"generator": "rest_api", "model": "m.json", "output_dir": "out", '
                  '"options": ["http_methods=GET, POST"]} keeps edits'],
        "results": [{"errors": ["pass `force: true` to overwrite, `dry_run: true` and `strict: true`"]}],
        "generators": [{"description": "keeps --force as it is"}],
    }


def test_for_client_rewrites_only_paths_that_start_at_the_root():
    root = Path("/app")
    output = {
        "files": ["/app/out/frontend/src/app/page.tsx"], "output_dir": "/app",
        "errors": ["/data/app is a file", "model '/app/m.json' is bad"],
        "notes": ["see https://example.org/app/docs", "/app~1/x is another folder"],
    }

    assert for_client(output, root) == {
        "files": ["out/frontend/src/app/page.tsx"], "output_dir": ".",
        "errors": ["/data/app is a file", "model 'm.json' is bad"],
        "notes": ["see https://example.org/app/docs", "/app~1/x is another folder"],
    }


def test_new_and_schema_start_a_model_under_the_root(serve, tmp_path):
    server = serve()

    written = server.call("new", path="models/machine.json", kind="state_machine")
    described = server.call("schema", topic="state_machine")
    outside = server.call("new", path="../machine.json")

    assert written["ok"] and written["files"] == ["models/machine.json"]
    assert server.call("validate", paths=["models/machine.json"], strict=True)["ok"]
    assert "StateMachine" in [cls["name"] for cls in described["description"]["classes"]]
    assert not outside["ok"] and "outside the server's root" in outside["errors"][0]
    assert not (tmp_path.parent / "machine.json").exists()


def test_convert_imports_another_notation(serve, write):
    server = serve()
    write("library.plantuml", "@startuml\nclass Book {\n  + title: str\n}\n@enduml\n")

    converted = server.call("convert", input="library.plantuml", output="library.json")

    assert converted["ok"] and converted["models"] == [{"name": "DomainModel", "kind": "domain", "type": "DomainModel"}]
    assert server.call("info", model="library.json")["models"][0]["classes"][0]["name"] == "Book"
