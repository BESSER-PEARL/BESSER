"""What a run builds: decided by the Phase 1 selector's LLM call.

Users name what they want ("a mobile app", "a Windows 11 app", "a CLI"),
not a framework. Every web-specific gate (the React CRUD rule, the
frontend checklist task, the missing-frontend blocker, the HTTP-write
contract, the server manifests of Phase 0.5) applies only to a
``web_app`` target. ``None`` means the decision failed; callers then keep
the pre-target behaviour. A modify run keeps its seed's target unless the
request names another kind (``pipeline/modify_run.py``).
"""
from __future__ import annotations

import re
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

# The one structured part of the request modeling-agent sends: for a non-web
# target whose description the instructions lack, a line of its own, anywhere
# in them (generation_handler._instructions_naming_target). Optional: an
# oversized spec falls back to the original request without it.
TARGET_LINE_RE = re.compile(r"^Target: (?P<description>.+?) \(not a web app\)\.", re.MULTILINE)

# Words that may name a non-web target. Only a cue to ask the selector again:
# "a Windows server" or "cancellation windows" name no target.
_OTHER_KIND_RE = re.compile(
    r"\b(?:native|mobile|android|ios|iphone|ipad|desktop|windows|macos|linux|"
    r"electron|tauri|wpf|cli|command[ -]line|game|extension|embedded|arduino|bot)\b",
    re.IGNORECASE)


def named_target(instructions: str | None) -> str | None:
    """The description on modeling-agent's ``Target:`` line, or ``None``."""
    match = TARGET_LINE_RE.search(instructions or "")
    return match.group("description").strip() if match else None


def may_name_other_target(instructions: str | None) -> bool:
    """True when the request may ask for a non-web target."""
    return named_target(instructions) is not None or bool(_OTHER_KIND_RE.search(instructions or ""))


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
