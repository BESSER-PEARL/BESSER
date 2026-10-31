"""The frontend import scan reads source with its comments removed.

A regex stripper took ``/*`` inside a string for a comment opener and
swallowed everything up to a later ``*/``, imports included, and cut a line
at a ``//`` inside a template literal (review 2026-10-30, R6,
``js_comment_strip.py``). An import the scan never sees is never checked.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.frontend_resolution import _SPEC_RE, _strip_comments


def _specs(source: str, name: str = "App.tsx") -> list[str]:
    return [m.group(2) for m in _SPEC_RE.finditer(_strip_comments(source, name))]


@pytest.mark.parametrize("source,spec", [
    ('const routes = "/api/*";\nimport OrderView from "./pages/OrderView";\nconst x = 1; /* done */\n',
     "./pages/OrderView"),
    ('const re = /https?:\\/\\//;\nimport api from "./api";\n', "./api"),
    ('const u = `${base}//v1`; import Cart from "./Cart";\n', "./Cart"),
])
def test_comment_markers_inside_literals_do_not_hide_imports(source, spec):
    assert spec in _specs(source)
    assert spec in _specs(source, "App.jsx")


def test_real_comments_are_removed_and_lines_kept():
    source = '// import Old from "./Old";\n/* import X from "./X";\n */\nimport New from "./New";\n'
    stripped = _strip_comments(source, "main.js")
    assert _specs(source, "main.js") == ["./New"]
    assert stripped.count("\n") == source.count("\n")


def test_unknown_config_file_is_read_as_javascript():
    assert "plugin-react" not in _strip_comments('{ // "plugin-react"\n}', ".babelrc")


@pytest.mark.parametrize("name", ["A.jsx", "A.tsx"])
def test_a_url_in_jsx_text_is_not_a_comment(name):
    """Review of #705: the lexer read "https://x.io" in JSX text as a line
    comment and dropped the rest of the line, a form included."""
    source = "const A = () => <div><p>See https://x.io</p><form onSubmit={save}></form></div>;\n"
    assert "onSubmit={save}" in _strip_comments(source, name)


def test_each_file_version_is_lexed_once(tmp_path, monkeypatch):
    import os

    import besser.spec_driven_agent.validation.frontend_resolution as resolution

    calls = []
    real = resolution._strip_comments
    monkeypatch.setattr(resolution, "_strip_comments", lambda text, name: calls.append(name) or real(text, name))
    resolution._stripped_file.cache_clear()
    source = tmp_path / "App.jsx"
    source.write_text("import a from './a';\n", encoding="utf-8")

    assert resolution._stripped_source(str(source)) == resolution._stripped_source(str(source))
    assert calls == ["App.jsx"]
    source.write_text("import b from './b'; // changed\n", encoding="utf-8")
    os.utime(source, ns=(1, 2_000_000_000))
    assert "./b" in resolution._stripped_source(str(source))
    assert calls == ["App.jsx", "App.jsx"]
