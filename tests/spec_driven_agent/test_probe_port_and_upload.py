"""`.besser_probe.py` must not make the agent fight its own helper.

A restart (`down`, `up`) moved the server to a new port every time, so a
command or script written with the earlier port hit "connection refused";
and `req` could send only JSON, so an upload endpoint had to be tested with
hand-written curl against that moving port.
"""
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, PROBE_SCRIPT, install_probe


def workspace(tmp_path):
    (tmp_path / "main_api.py").write_text("", encoding="utf-8")
    (tmp_path / "sql_alchemy.py").write_text("", encoding="utf-8")
    assert install_probe(str(tmp_path))
    return tmp_path


def load_probe(tmp_path):
    namespace = {"__file__": str(tmp_path / PROBE_FILENAME), "__name__": "besser_probe_under_test"}
    exec(compile(PROBE_SCRIPT, PROBE_FILENAME, "exec"), namespace)
    return namespace


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fake_boot(ns, new_port):
    """Boot without a real server: the child stays alive and answers at once."""
    class Child:
        pid = 4242

        def __init__(self, argv, **_kwargs):
            pass

        def poll(self):
            return None

    ns.update(free_port=lambda: new_port, is_up=lambda port: True,
              subprocess=SimpleNamespace(Popen=Child, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT,
                                         CREATE_NEW_PROCESS_GROUP=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                                         run=subprocess.run))


def test_a_restart_keeps_the_port_the_server_had(tmp_path):
    workspace(tmp_path)
    port = unused_port()
    server = subprocess.Popen([sys.executable, "-c",
                               "# BESSER_PROBE_SERVE port=%d\nimport time; time.sleep(60)" % port])
    try:
        (tmp_path / ".besser_probe.state.json").write_text(
            json.dumps({"pid": server.pid, "port": port, "backend": "."}), encoding="utf-8")
        down = subprocess.run([sys.executable, PROBE_FILENAME, "down"], cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)
        assert "STOPPED" in down.stdout, down.stdout + down.stderr
    finally:
        if server.poll() is None:
            server.kill()
        server.wait()

    ns = load_probe(tmp_path)
    fake_boot(ns, new_port=port + 1 if port < 65535 else port - 1)
    assert ns["boot"](quiet=True) == (0, port)


def test_a_port_someone_else_now_holds_is_not_taken_again(tmp_path):
    workspace(tmp_path)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        taken = holder.getsockname()[1]
        (tmp_path / ".besser_probe.state.json").write_text(json.dumps({"port": taken}), encoding="utf-8")
        ns = load_probe(tmp_path)
        fresh = unused_port()
        fake_boot(ns, new_port=fresh)
        assert ns["boot"](quiet=True) == (0, fresh)


def capture_server():
    """A local server that records one POST and answers 200."""
    import http.server
    import threading

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["type"] = self.headers["Content-Type"]
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def form_parts(content_type, body):
    from email.parser import BytesParser
    from email.policy import HTTP
    message = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    return {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}


def test_req_sends_files_and_fields_as_multipart_form_data(tmp_path):
    workspace(tmp_path)
    (tmp_path / "orders.csv").write_bytes(b"id,total\n1,9.5\n")
    ns = load_probe(tmp_path)
    server, seen = capture_server()
    try:
        ns.update(ensure_up=lambda: server.server_address[1])
        code = ns["cmd_req"](["POST", "/order/import", "-F", "file=@%s;type=text/csv" % (tmp_path / "orders.csv"),
                              "-F", "'note=first batch'"])
    finally:
        server.shutdown()
        server.server_close()

    assert code == 0
    assert seen["type"].startswith("multipart/form-data; boundary=")
    parts = form_parts(seen["type"], seen["body"])
    assert parts["file"].get_filename() == "orders.csv"
    assert parts["file"].get_content_type() == "text/csv"
    assert parts["file"].get_payload(decode=True) == b"id,total\n1,9.5\n"
    assert parts["note"].get_payload(decode=True) == b"first batch"


def test_req_names_a_file_it_cannot_read(tmp_path, capsys):
    workspace(tmp_path)
    ns = load_probe(tmp_path)
    ns.update(ensure_up=lambda: 1 / 0)
    assert ns["cmd_req"](["POST", "/order/import", "-F", "file=@missing.pdf"]) == 2
    assert "BAD_FORM" in capsys.readouterr().out


# --- a JSON body the shell cannot mangle ------------------------------------

def _sent_body(tmp_path, argv, monkeypatch=None, stdin=None):
    workspace(tmp_path)
    ns = load_probe(tmp_path)
    seen = {}

    def request(method, path, body, port, **_kwargs):
        seen["body"] = body
        return 201, "{}"

    if stdin is not None:
        import io
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    ns.update(ensure_up=lambda: 5001, request=request)
    return ns["cmd_req"](argv), seen.get("body")


def test_req_reads_the_body_from_stdin(tmp_path, monkeypatch):
    code, body = _sent_body(tmp_path, ["POST", "/book", "-"], monkeypatch,
                            stdin='{"title": "It\'s here", "pages": 3}\n')
    assert (code, body) == (0, {"title": "It's here", "pages": 3})


def test_req_reads_the_body_from_a_file(tmp_path):
    (tmp_path / "book.json").write_text('{"title": "x"}', encoding="utf-8")
    code, body = _sent_body(tmp_path, ["POST", "/book", "@%s" % (tmp_path / "book.json")])
    assert (code, body) == (0, {"title": "x"})


def test_a_body_whose_quotes_the_shell_removed_gets_the_quoting_advice(tmp_path, capsys):
    # bash turns  req POST /book {"title":"Clean Code"}  into these arguments.
    code, body = _sent_body(tmp_path, ["POST", "/book", "{title:Clean", "Code}"])
    out = capsys.readouterr().out
    assert code == 2 and body is None
    assert "single quotes" in out and "stdin" in out


def test_an_unknown_option_is_named_instead_of_read_as_json(tmp_path, capsys):
    code, _ = _sent_body(tmp_path, ["POST", "/book", '{"title": ""}', "--expect", "400"])
    out = capsys.readouterr().out
    assert code == 2
    assert "--expect" in out and "BAD_JSON" not in out


def test_the_runbook_quotes_every_json_body_it_shows(tmp_path):
    """Bash strips the double quotes of an unquoted body, so a copied example must quote it."""
    import re

    from besser.spec_driven_agent.agent.runbook import runbook_section

    section = runbook_section(str(workspace(tmp_path)))
    bodies = re.findall(r"req \w+ \S+ (\S*\{\S*)", section)
    assert bodies and all(body.startswith("'") or body == "{}" for body in bodies), bodies


def test_req_with_an_empty_stdin_is_refused_not_sent_as_an_empty_object(tmp_path, monkeypatch, capsys):
    code, body = _sent_body(tmp_path, ["POST", "/book", "-"], monkeypatch, stdin="")
    assert (code, body) == (2, None)
    assert "BAD_BODY" in capsys.readouterr().out


def test_form_values_keep_their_own_quotes_and_options(tmp_path):
    workspace(tmp_path)
    (tmp_path / "cover.bin").write_bytes(b"\x00\x01")
    ns = load_probe(tmp_path)
    rest, fields = ns["split_form"](["-F", "'title=\"x\"'", "-F", 'note=say "hi"',
                                     "-F", "file=@%s;type=image/png;filename=front.png" % (tmp_path / "cover.bin")])
    assert rest == []
    assert fields[:2] == [("title", '"x"'), ("note", 'say "hi"')]

    body, content_type = ns["encode_form"](fields + [('we"ird', "v")])
    parts = form_parts(content_type, body)
    assert parts["title"].get_payload(decode=True) == b'"x"'
    assert parts["note"].get_payload(decode=True) == b'say "hi"'
    assert parts["file"].get_filename() == "front.png"
    assert parts["file"].get_content_type() == "image/png"
    assert parts["file"].get_payload(decode=True) == b"\x00\x01"
    assert b'name="we%22ird"' in body
