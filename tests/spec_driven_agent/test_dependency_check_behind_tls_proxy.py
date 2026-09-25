"""The pip dependency check behind a TLS-inspecting proxy.

All four measured runs on a LIST laptop reported the same blocker:

    Dependency conflict in web_app/backend/requirements.txt:
    ERROR: No matching distribution found for anyio<4.0.0,>=3.7.1
    [notice] A new release of pip is available: 24.0 -> 26.2.1

There was no conflict. ``_safe_subprocess_env`` stripped the CA-bundle
variables ("CERT" is a deny substring), so pip could not verify the proxy's
certificate, and the cause sat in the WARNING lines the excerpt dropped while
two ``[notice]`` lines filled it. ``PIP_STDERR`` is pip's real output from that
laptop with the pre-fix environment.
"""
import os
import subprocess

from besser.spec_driven_agent.execution.process import _safe_subprocess_env
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator, _classify_issue

_SSL = ("WARNING: Retrying (Retry(total={n}, connect=None, read=None, redirect=None, "
        "status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, "
        "'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed "
        "certificate in certificate chain (_ssl.c:1006)'))': /simple/anyio/")
PIP_STDERR = "\n".join([
    *(_SSL.format(n=n) for n in (4, 3, 2, 1, 0)),
    "ERROR: Could not find a version that satisfies the requirement anyio<4.0.0,>=3.7.1 "
    "(from versions: none)",
    "ERROR: No matching distribution found for anyio<4.0.0,>=3.7.1",
    "",
    "[notice] A new release of pip is available: 24.0 -> 26.2.1",
    "[notice] To update, run: python.exe -m pip install --upgrade pip",
    "",
])

CONFLICT_STDERR = "\n".join([
    "ERROR: Cannot install fastapi==0.110.0 and starlette==0.20.0 because these "
    "package versions have conflicting dependencies.",
    "ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/",
    "",
    "[notice] A new release of pip is available: 24.0 -> 26.2.1",
    "[notice] To update, run: python.exe -m pip install --upgrade pip",
])


# --------------------------------------------------------------------------- #
# The environment
# --------------------------------------------------------------------------- #
def test_ca_bundle_variables_reach_the_subprocess(monkeypatch, tmp_path):
    bundle = tmp_path / "corp_ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    names = ("NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
             "PIP_CERT", "CURL_CA_BUNDLE", "NPM_CONFIG_CAFILE")
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin", **{n: str(bundle) for n in names}})

    env = _safe_subprocess_env()

    for name in names:
        assert env.get(name) == str(bundle), f"{name} was stripped; pip/npm cannot verify TLS"


def test_secrets_are_still_stripped_alongside_the_ca_bundle(monkeypatch, tmp_path):
    """Passing trust-store paths must not open the deny rule."""
    bundle = tmp_path / "corp_ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(os, "environ", {
        "PATH": "/usr/bin",
        "SSL_CERT_FILE": str(bundle),
        "ANTHROPIC_API_KEY": "sk-secret",
        "OPENAI_API_KEY": "sk-secret",
        "NPM_AUTH_TOKEN": "npm_x",
        "GITHUB_TOKEN": "ghp_x",
        "CLIENT_CERT_PASSWORD": "hunter2",
        "SSL_CLIENT_CERT_KEY": str(bundle),  # a key path, not a trust store
        # A CA-bundle NAME holding something other than a path is dropped.
        "REQUESTS_CA_BUNDLE": "-----BEGIN PRIVATE KEY-----MIIE",
    })

    env = {name.upper() for name in _safe_subprocess_env()}

    assert "SSL_CERT_FILE" in env
    for banned in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NPM_AUTH_TOKEN", "GITHUB_TOKEN",
                   "CLIENT_CERT_PASSWORD", "SSL_CLIENT_CERT_KEY", "REQUESTS_CA_BUNDLE"):
        assert banned not in env, f"{banned} reached an LLM-invoked subprocess"


# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #
def _dependency_findings(model, tmp_path, monkeypatch, stderr):
    """Run the real Phase 3 collector with pip answering ``stderr``."""
    backend = tmp_path / "web_app" / "backend"
    backend.mkdir(parents=True)
    (backend / "requirements.txt").write_text("anyio<4.0.0,>=3.7.1\n")

    class _Client:
        model = "mock-model"

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    from besser.spec_driven_agent.providers.llm_client import UsageTracker
    _Client.usage = UsageTracker("mock-model")
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=model, output_dir=str(tmp_path),
        allow_shell_tools=True, enable_toolchain_validation=False,
        enable_checkpointing=False,
    )
    for name in ("_collect_frontend_contract_issues", "_collect_ruff_issues",
                 "_collect_execution_issues", "_collect_tsc_issues",
                 "_collect_requirement_issues", "_collect_task_issues",
                 "_collect_data_contract_issues", "_collect_missing_frontend_issue",
                 "_collect_framework_switch_issues"):
        monkeypatch.setattr(orch, name, lambda: [])

    def pip(command, *args, **kwargs):
        if "pip" in command:
            return subprocess.CompletedProcess(command, 1, "", stderr)
        return subprocess.CompletedProcess(command, 0, "", "")

    # Whichever launcher the collector uses.
    monkeypatch.setattr(subprocess, "run", pip)
    monkeypatch.setattr(orchestrator_module, "run_bounded", pip, raising=False)

    return [i for i in orch._collect_validation_issues()
            if "requirements.txt" in i.message]


def test_a_tls_failure_is_a_warning_not_a_dependency_conflict(
        simple_library_book_model, tmp_path, monkeypatch):
    findings = _dependency_findings(simple_library_book_model, tmp_path, monkeypatch, PIP_STDERR)

    assert len(findings) == 1, findings
    finding = findings[0]
    assert finding.severity == "warning", finding
    assert "Dependency conflict" not in finding.message
    assert "could not verify dependencies" in finding.message
    assert "network/TLS error" in finding.message
    assert "CERTIFICATE_VERIFY_FAILED" in finding.message, "the cause must be named"


def test_a_real_conflict_is_still_a_blocker_and_names_the_error(
        simple_library_book_model, tmp_path, monkeypatch):
    findings = _dependency_findings(simple_library_book_model, tmp_path, monkeypatch,
                                    CONFLICT_STDERR)

    assert len(findings) == 1, findings
    finding = findings[0]
    assert finding.severity == "blocker"
    assert finding.message.startswith("Dependency conflict in web_app/backend/requirements.txt")
    assert "ResolutionImpossible" in finding.message
    assert "conflicting dependencies" in finding.message
    assert "[notice]" not in finding.message, "pip's upgrade nag hid the real error"


def test_the_network_finding_classifies_as_a_warning_on_its_own():
    from besser.spec_driven_agent.validation.issues import dependency_check_issue

    message = dependency_check_issue("backend/requirements.txt", PIP_STDERR)

    assert _classify_issue(message).severity == "warning"
