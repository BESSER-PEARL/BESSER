"""What a run builds: decided by the Phase 1 selector's LLM call.

Users name what they want ("a mobile app", "a Windows 11 app", "a CLI"),
not a framework. Every web-specific gate (the React CRUD rule, the
frontend checklist task, the missing-frontend blocker, the HTTP-write
contract, the server manifests of Phase 0.5) applies only to a
``web_app`` target. ``None`` means the decision failed; callers then keep
the pre-target behaviour. A modify run asks the same call whether the
request changes the existing app's target (``pipeline/modify_run.py``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

WEB_APP = "web_app"
BACKEND = "backend"
OTHER = "other"
TARGET_KINDS = (WEB_APP, BACKEND, OTHER)

# Shared by the forced selector tool and the recipe, so both carry one shape.
TARGET_TOOL_PROPERTIES = {
    "target_kind": {"type": "string", "enum": list(TARGET_KINDS)},
    "target_description": {
        "type": "string",
        "description": "What will be built, in a few words, e.g. 'native Android/iOS app'.",
    },
    "has_user_interface": {"type": "boolean"},
}

TARGET_RULES = """\
TARGET — decide WHAT the user wants built, from their words:
- web_app: anything used in a browser: web app, website, dashboard, portal, PWA,
  a "mobile-friendly" or "responsive" web app. Also a bare "app" or "application"
  with no platform named (the default).
- backend: API-only / headless: a REST API, a backend, a server, also "an API
  for my mobile app".
- other: anything that is not a web app or a backend: a native mobile app
  (Android / iOS / "mobile app"), a desktop app (Windows / macOS / Linux,
  "local", Electron, Tauri, WPF), a CLI, a game, a browser or IDE extension,
  embedded / Arduino code, a bot, a library, plain classes.
Platform words that are not the target do not count: "deploy on a Windows
server", "cancellation windows", "native speakers".
target_description names the product in a few words ("native Android/iOS app",
"Windows 11 desktop app", "Python CLI"), with the framework only if the user named one.
has_user_interface: true when people use the result through screens (web, mobile,
desktop, game), false for APIs, CLIs, libraries and classes."""

# A NONE stands only for a named stack: the rules give every Python web app or
# backend a generator, so a NONE without one keeps the keyword fallback.
STACK_TOOL_PROPERTIES = {
    "named_stack": {
        "type": "string",
        "description": ("The programming language or framework the user asks the software to be "
                        "built with, e.g. 'Rust + Axum', 'Flutter', 'Next.js', 'Django'; '' when "
                        "the user names none."),
    },
}

STACK_RULES = """\
named_stack: only a language or framework the user asks the software to be built
with. Words of the domain are not stacks, even when they spell one: "an express
checkout", "a Java coffee shop", "rust-proof paint", "Go-karts", "the key of F#"."""

# Asked only on a modify run, which changes an app that already exists.
MODIFY_TOOL_PROPERTIES = {
    "changes_target": {"type": "boolean"},
    "is_failure_report": {"type": "boolean"},
    "reported_entity": {"type": "string"},
}

MODIFY_RULES = """\
EXISTING APP: this request changes an app that already exists.
- changes_target: true only when the request asks for a different kind of software
  ("turn it into a mobile app", "I only need the API now", "make it a CLI") or for
  another stack ("rewrite it in Vue"). New features, screens or fixes keep the target,
  and so does naming the same target in other words.
- is_failure_report: true when the newest user message says the existing app
  misbehaves: an error, a crash, a wrong result, a page or control that does not do
  what it should. False for a new feature or change, even one that names the errors,
  exceptions or status codes the app should produce ("return 409 on a duplicate",
  "log every IntegrityError").
- reported_entity: for a failure report, the model class it is about (one of the
  model classes below), else a short noun for the feature; '' otherwise."""


@dataclass(frozen=True)
class Target:
    kind: str
    description: str
    has_user_interface: bool

    @property
    def is_web_app(self) -> bool:
        return self.kind == WEB_APP

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload) -> Target | None:
        """Parse a selector tool payload or a recipe entry; ``None`` when unusable."""
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind", payload.get("target_kind"))
        if kind not in TARGET_KINDS:
            return None
        description = str(payload.get("description", payload.get("target_description")) or "").strip()
        has_ui = payload.get("has_user_interface")
        if not isinstance(has_ui, bool):
            has_ui = kind == WEB_APP
        return cls(kind=kind, description=description[:120], has_user_interface=has_ui)


@dataclass(frozen=True)
class Routing:
    """One answer of the forced ``choose_generator`` call."""

    target: Target | None
    generator: str  # a generator name, "NONE", or "" when the answer has none
    named_stack: str = ""
    changes_target: bool = False
    is_failure_report: bool = False
    reported_entity: str = ""

    @classmethod
    def from_payload(cls, payload) -> Routing:
        payload = payload if isinstance(payload, dict) else {}

        def _text(key: str) -> str:
            return str(payload.get(key) or "").strip()[:120]

        def _flag(key: str) -> bool:
            # Some providers send a forced tool's boolean as a string.
            return str(payload.get(key)).lower() == "true"

        return cls(
            target=Target.from_payload(payload),
            generator=_text("generator"),
            named_stack=_text("named_stack"),
            changes_target=_flag("changes_target"),
            is_failure_report=_flag("is_failure_report"),
            reported_entity=_text("reported_entity"),
        )
