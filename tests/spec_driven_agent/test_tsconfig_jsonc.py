"""tsconfig and jsconfig files are JSONC: comments and trailing commas are valid.

The write diagnostic parsed them as strict JSON, so the ``tsconfig.app.json``
that ``npm create vite@latest -- --template react-ts`` writes was reported as
a parse error (review 2026-10-30, R7, ``tsconfig_jsonc.py``).
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.write_diagnostics import diagnose_written_content

VITE_TSCONFIG_APP = """{
  "compilerOptions": {
    "target": "ES2022",
    "jsx": "react-jsx",

    /* Bundler mode */
    "moduleResolution": "bundler",
    "noEmit": true,

    // Linting
    "strict": true,
  },
  "include": ["src",],
  "exclude": ["a,]"]
}
"""


@pytest.mark.parametrize("path", [
    "frontend/tsconfig.app.json", "tsconfig.json", "web/jsconfig.json", ".vscode/settings.json",
    # Other configs whose tools read them as JSONC / JSON5.
    "frontend/.eslintrc.json", ".devcontainer/devcontainer.json", ".devcontainer.json",
    "frontend/.babelrc.json", "frontend/babel.config.json", "api-extractor.json",
    "frontend/.vscode/launch.json", "settings.jsonc",
])
def test_jsonc_config_is_not_a_parse_error(path):
    assert diagnose_written_content(path, VITE_TSCONFIG_APP) == []


def test_plain_json_stays_strict():
    findings = diagnose_written_content("frontend/package.json", '{"name": "app",}\n')
    assert findings and findings[0]["code"] == "parse"


def test_a_broken_jsonc_file_is_reported():
    findings = diagnose_written_content("frontend/.eslintrc.json", '{\n  // rules\n  "extends": [\n}\n')
    assert findings and findings[0]["code"] == "parse"
    assert diagnose_written_content("deno.jsonc", '{\n  // x\n  "tasks": \n}\n')[0]["code"] == "parse"


def test_broken_tsconfig_is_still_reported_on_its_line():
    findings = diagnose_written_content("tsconfig.json", '{\n  // ok\n  "strict": true\n  "jsx": "react"\n}\n')
    assert findings and findings[0]["line"] == 4
