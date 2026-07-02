"""
Phase 0.5 — pre-generated stack metadata.

BESSER's deterministic Phase 1 generators only cover Python stacks
(Django, FastAPI, SQLAlchemy, Pydantic, plain Python classes). For
non-Python targets — Next.js / TypeScript, Rust / Cargo, Kotlin /
Gradle — Phase 1 is a no-op and the customise loop is expected to
emit *every* file from scratch, including the project's build-metadata
file (tsconfig.json, Cargo.toml, build.gradle.kts, …).

Two problems with that approach:

1. The customise LLM occasionally forgets one of these files (no
   tsconfig.json, no Cargo.toml). The artifact then fails the bench's
   per-project compile check 0/n even when the source files themselves
   are correct.
2. When the LLM does emit the metadata, it often invents dependency
   versions that don't exist on the registry.

Phase 0.5 fixes both by pre-creating a *minimal but valid* manifest
for stacks BESSER doesn't have a deterministic generator for. The
customise loop sees these files in the inventory and is told to build
on top of them rather than rewrite them. The result is a reliable
floor for project-level toolchain checks.

The templates are deliberately stack-neutral — they don't bake in
specific business choices the customise loop owns (no entity-specific
dependencies, no opinionated source structure). They DO commit to:

- Concrete dependency versions that exist on the public registry
  at the time of writing.
- Idiomatic defaults for the target stack (e.g. Node 18 + ES2022 +
  bundler resolution for TypeScript, edition 2021 for Rust, JDK 17
  for Kotlin / Spring Boot 3).
- File names and locations the toolchain expects in the default
  layout (tsconfig.json at the project root, Cargo.toml at the
  project root, settings.gradle.kts next to build.gradle.kts).

The detection heuristic is a deliberately cheap substring match.
Adding a richer parser would be overkill given the bench's
instruction format ("Build a Next.js …", "Build a Rust …").
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterable

logger = logging.getLogger(__name__)


# Order matters: a request mentioning both "Next.js" and "TypeScript"
# should resolve to ``nextjs`` (the more specific stack). The first
# match wins; everything below is checked top-down.
_STACK_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Next.js absorbs TypeScript when both are mentioned.
    ("nextjs", ("next.js", "nextjs", "next js")),
    # Kotlin / Spring Boot bundle together — kotlinc + Gradle.
    ("kotlin_spring", ("kotlin", "spring boot", "springboot", "spring-boot")),
    # Rust + Cargo. ``axum``/``actix``/``rocket`` imply Rust without
    # the user spelling out "Rust", so they map to the same template.
    ("rust", ("rust", "cargo", "axum", "actix", "rocket")),
)


# Stacks where BESSER already has a deterministic Phase 1 generator.
# Phase 0.5 MUST NOT pre-create files for these — the generator owns
# the manifest (pyproject.toml, requirements.txt, …). The set is keyed
# on instruction substrings the orchestrator's existing keyword
# selector also recognises.
_PYTHON_STACK_KEYWORDS: frozenset[str] = frozenset(
    {
        "django",
        "fastapi",
        "fast api",
        "flask",
        "pydantic",
        "sqlalchemy",
    }
)


def _contains_word(text: str, needle: str) -> bool:
    """Case-insensitive substring match with word-boundary fallback.

    Falls back to plain substring when the needle contains punctuation
    that ``\b`` would treat as a non-word character (``next.js``).
    """
    lowered = text.lower()
    needle = needle.lower()
    if needle.replace(".", "").replace(" ", "").replace("-", "").isalnum():
        # Pure alphanumeric — use a word boundary so "rust" doesn't
        # match "trustworthy".
        return bool(re.search(r"\b" + re.escape(needle) + r"\b", lowered))
    return needle in lowered


def detect_stack(instructions: str) -> str | None:
    """Detect the target stack from natural-language instructions.

    Returns one of:
        ``"nextjs"`` | ``"rust"`` | ``"kotlin_spring"`` | ``None``

    ``None`` means either:
      - BESSER already has a Phase 1 generator for the stack (Python
        family) — Phase 0.5 must not interfere.
      - The instructions don't mention a stack we have a template for.

    The check for the Python family is a hard early-out: even if a
    Python instruction happens to contain the word "rust" in some
    other context, Phase 0.5 stays out of the way.
    """
    if not instructions or not instructions.strip():
        return None

    # Python-family early-out: BESSER's Phase 1 generators own these
    # stacks. Phase 0.5 must not create a tsconfig/Cargo.toml just
    # because the user mentioned "Next.js was an inspiration but build
    # me a FastAPI app".
    for kw in _PYTHON_STACK_KEYWORDS:
        if _contains_word(instructions, kw):
            return None

    for stack_id, needles in _STACK_KEYWORDS:
        for needle in needles:
            if _contains_word(instructions, needle):
                return stack_id

    return None


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
#
# Each template is a dict mapping relative file paths to file contents.
# Contents are stored as plain strings — we deliberately avoid Jinja2
# rendering here because Phase 0.5 outputs are stack-neutral by design
# (no model-specific substitutions). Keeping the templates as literal
# strings means a developer can copy/paste them straight out of this
# module to verify they're valid.
#
# Versions referenced below are the latest stable as of 2026-Q2 and
# are known to coexist; bumping them is fine as long as the bumped
# combination still resolves on its registry.

_TSCONFIG_NEXTJS = """\
{
  "compilerOptions": {
    "target": "es2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      { "name": "next" }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
"""

_NEXT_ENV_DTS = """\
/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/basic-features/typescript for more information.
"""

_PACKAGE_JSON_NEXTJS = """\
{
  "name": "nextjs-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/node": "^20.14.10",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "typescript": "^5.5.3"
  }
}
"""


_CARGO_TOML_RUST = """\
[package]
name = "app"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.39", features = ["full"] }
axum = "0.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
"""


_BUILD_GRADLE_KOTLIN = """\
plugins {
    kotlin("jvm") version "1.9.24"
    kotlin("plugin.spring") version "1.9.24"
    kotlin("plugin.jpa") version "1.9.24"
    id("org.springframework.boot") version "3.3.2"
    id("io.spring.dependency-management") version "1.1.6"
}

group = "com.example"
version = "0.0.1-SNAPSHOT"

java {
    sourceCompatibility = JavaVersion.VERSION_17
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin")
    implementation("org.jetbrains.kotlin:kotlin-reflect")
    runtimeOnly("com.h2database:h2")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        freeCompilerArgs = listOf("-Xjsr305=strict")
        jvmTarget = "17"
    }
}

tasks.withType<Test> {
    useJUnitPlatform()
}
"""


_SETTINGS_GRADLE_KOTLIN = """\
rootProject.name = "app"
"""


# Per-stack file map. Each entry is a list of (relative_path, content)
# tuples so iteration is deterministic — important for tests.
_STACK_FILES: dict[str, tuple[tuple[str, str], ...]] = {
    "nextjs": (
        ("tsconfig.json", _TSCONFIG_NEXTJS),
        ("next-env.d.ts", _NEXT_ENV_DTS),
        ("package.json", _PACKAGE_JSON_NEXTJS),
    ),
    "rust": (
        ("Cargo.toml", _CARGO_TOML_RUST),
    ),
    "kotlin_spring": (
        ("build.gradle.kts", _BUILD_GRADLE_KOTLIN),
        ("settings.gradle.kts", _SETTINGS_GRADLE_KOTLIN),
    ),
}


# Friendly label per stack — used in logs and the LLM-facing inventory
# note so the customise loop knows what was pre-generated.
_STACK_LABEL: dict[str, str] = {
    "nextjs": "Next.js (TypeScript)",
    "rust": "Rust / Cargo",
    "kotlin_spring": "Kotlin / Spring Boot (Gradle)",
}


def stack_label(stack_id: str) -> str:
    """Human-readable name for a stack id."""
    return _STACK_LABEL.get(stack_id, stack_id)


def files_for(stack_id: str) -> tuple[tuple[str, str], ...]:
    """Public access to the file map — used by tests."""
    return _STACK_FILES.get(stack_id, ())


def pre_generate_metadata(stack_id: str, output_dir: str) -> list[str]:
    """Write Phase 0.5 metadata files for ``stack_id`` into ``output_dir``.

    Returns the list of files that were created (relative paths). If
    a file already exists at the target path — e.g. a previous run
    left it, or a deterministic generator emitted it earlier in the
    same process — it is left untouched and not included in the
    returned list. This keeps Phase 0.5 strictly additive.

    Validates JSON content before writing (catches typos in the
    templates above on first call rather than at the toolchain
    boundary).
    """
    files = _STACK_FILES.get(stack_id)
    if not files:
        logger.debug("Phase 0.5: no template for stack %s", stack_id)
        return []

    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for rel_path, content in files:
        # Validate JSON templates at write time so a malformed
        # template doesn't silently produce broken artifacts.
        if rel_path.endswith(".json"):
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:  # pragma: no cover - template bug
                raise RuntimeError(
                    f"Phase 0.5 template for {stack_id}/{rel_path} is not "
                    f"valid JSON: {exc}"
                ) from exc

        target = os.path.join(output_dir, rel_path)
        if os.path.exists(target):
            logger.debug(
                "Phase 0.5: %s already exists, leaving untouched", rel_path
            )
            continue
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(rel_path)

    if written:
        logger.info(
            "Phase 0.5: pre-generated %d %s metadata file(s): %s",
            len(written), stack_label(stack_id), ", ".join(written),
        )
    return written


def supported_stacks() -> Iterable[str]:
    """Stacks Phase 0.5 has templates for. Used by tests."""
    return tuple(_STACK_FILES.keys())


# ---------------------------------------------------------------------------
# Idiomatic-conventions prompt injection
# ---------------------------------------------------------------------------
#
# A vibe-bench run found LLM-generated code LESS idiomatic than a naive-LLM
# baseline in ~80% of scenarios: Spring Boot entities reaching for `Int`
# instead of `Long` ids, Flask/FastAPI storing money as `float`, axum
# handlers `unwrap()`-ing instead of returning `Result`, entities reused as
# request bodies instead of dedicated DTOs, .... The customise-loop prompt
# said nothing about stack-specific conventions, so the LLM defaulted to
# whatever's most common in its training data rather than what an
# experienced dev on that stack would write.
#
# This section adds a SHORT, targeted "idiomatic conventions" block to the
# system prompt when the instructions name a recognisable stack. It's
# guidance, not a spec: a handful of bullets, not a style guide -- long
# enough to redirect the model's defaults, short enough not to drown out
# the Rules above it for requests where it doesn't apply.
#
# Detection here is deliberately independent of ``detect_stack`` above:
# that function's Python-family early-out exists to protect Phase 0.5
# (don't pre-write a tsconfig.json for a FastAPI request) -- the opposite
# of what we want here, since Flask/FastAPI are exactly two of the stacks
# this guidance targets. It also isn't gated on "no deterministic
# generator ran": the Phase 2 LLM writes substantial code on top of the
# FastAPI/Django scaffold too (auth, business endpoints, ...), and that
# code benefits from the same reminders the from-scratch stacks get.

_IDIOM_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Spring Boot conventions apply whether the request says Java or Kotlin.
    ("spring_boot", ("spring boot", "springboot", "spring-boot")),
    # Rust web frameworks bundle together -- axum-specific bullets are a
    # minority of the block; the Result/Option guidance applies just as
    # well to actix/rocket handlers.
    ("rust_axum", ("rust", "cargo", "axum", "actix", "rocket")),
    ("python_web", ("flask", "fastapi", "fast api")),
)

# Stacks without a dedicated idiom block above, but common enough in
# free-form requests to warrant the generic fallback rather than silence.
# Limited to stacks the LLM writes largely from scratch (no BESSER
# deterministic generator already imposing structure).
_GENERIC_IDIOM_KEYWORDS: tuple[str, ...] = (
    "next.js", "nextjs", "next js", "express", "nestjs", "django",
    "laravel", "rails", "golang", "angular", "vue", "svelte", "nuxt",
    "asp.net", "dotnet", ".net", "csharp", "c#", "php", "ruby",
)


_IDIOM_BLOCKS: dict[str, str] = {
    "spring_boot": (
        "- Entity IDs are `Long` (boxed), never `Int`/`int` — matches "
        "JPA/Hibernate `@GeneratedValue` and avoids overflow on "
        "auto-increment PKs.\n"
        "- Signal errors with `ResponseStatusException` (or a "
        "`@ControllerAdvice` handler for a custom exception) — never let "
        "an unchecked exception fall through to a bare 500.\n"
        "- Keep the layers separate: `@RestController` → `@Service` → "
        "`@Repository`. No business logic in controllers, no `@Entity` "
        "returned directly from an endpoint.\n"
        "- Dedicated request/response DTOs (records or classes) at the "
        "controller boundary — never the `@Entity` itself. Validate DTOs "
        "with `jakarta.validation` annotations (`@NotNull`, `@Size`, …).\n"
        "- RESTful resource naming: plural nouns, nested paths for owned "
        "relationships (`/orders/{id}/items`), correct verb-to-status-code "
        "mapping (`POST` → 201, `DELETE` → 204)."
    ),
    "python_web": (
        "- Money/currency fields are `Decimal`, never `float` — floats "
        "silently corrupt totals.\n"
        "- Separate request schema, response schema, and ORM/DB model — "
        "don't reuse the SQLAlchemy (or Django) model as the request "
        "body.\n"
        "- `snake_case` for functions, variables, and JSON field names; "
        "`PascalCase` reserved for classes.\n"
        "- Raise `HTTPException` (FastAPI) or a registered error handler "
        "(Flask) with the correct status code — don't let a bare "
        "exception 500 or return 200 with an error payload.\n"
        "- RESTful resource naming: plural nouns (`/orders`, "
        "`/orders/{id}`), and type-hint every function signature."
    ),
    "rust_axum": (
        "- Update-DTO fields are `Option<T>` so \"not provided\" and "
        "\"set to null/default\" stay distinguishable on a PATCH.\n"
        "- Handlers return `Result<T, AppError>` with `AppError` "
        "implementing `IntoResponse` — no `unwrap()`/`expect()` on "
        "request-derived data.\n"
        "- Separate `#[derive(Deserialize)]` request structs and "
        "`#[derive(Serialize)]` response structs from the internal "
        "domain struct.\n"
        "- `snake_case` for fields/functions, `PascalCase` for "
        "structs/enums (rustfmt defaults, not optional style).\n"
        "- Map `AppError` variants to real HTTP status codes — avoid a "
        "single catch-all 500."
    ),
    "generic": (
        "- Separate request/response DTOs from the persistence/domain "
        "model — never expose the ORM entity directly through the API.\n"
        "- RESTful resource naming: plural nouns, nested paths for owned "
        "relationships, correct HTTP verb-to-status-code mapping.\n"
        "- Use the target language's idiomatic casing consistently "
        "(`snake_case`, `camelCase`, or `PascalCase` — whichever that "
        "language/framework expects) rather than mixing conventions.\n"
        "- Centralised error handling with real status codes — no bare "
        "500s or silently swallowed exceptions.\n"
        "- Validate input at the DTO/schema boundary, not with scattered "
        "manual `if` checks deeper in the code."
    ),
}

_IDIOM_STACK_LABEL: dict[str, str] = {
    "spring_boot": "Spring Boot",
    "python_web": "Flask / FastAPI (Python)",
    "rust_axum": "Rust (axum)",
    "generic": "this stack",
}


def detect_idiom_stack(instructions: str) -> str | None:
    """Identify which idiom block (if any) applies to ``instructions``.

    Returns one of ``"spring_boot"``, ``"python_web"``, ``"rust_axum"``,
    ``"generic"``, or ``None`` when no recognisable stack is named.

    See the module comment above ``_IDIOM_KEYWORDS`` for why this is a
    separate detector from ``detect_stack`` rather than a reuse of it.
    """
    if not instructions or not instructions.strip():
        return None

    for idiom_id, needles in _IDIOM_KEYWORDS:
        for needle in needles:
            if _contains_word(instructions, needle):
                return idiom_id

    for needle in _GENERIC_IDIOM_KEYWORDS:
        if _contains_word(instructions, needle):
            return "generic"

    return None


def idiom_guidance_section(instructions: str) -> str:
    """Render the "idiomatic conventions" prompt block for ``instructions``.

    Returns ``""`` when no target stack is recognised — callers should
    not emit an empty heading in that case. Ends with a blank line (two
    trailing newlines) when non-empty so it composes cleanly between two
    ``##`` sections in the caller's prompt template; returns "" (no
    newlines at all) when nothing was detected so it's a true no-op.
    """
    idiom_id = detect_idiom_stack(instructions)
    if idiom_id is None:
        return ""

    label = _IDIOM_STACK_LABEL.get(idiom_id, idiom_id)
    bullets = _IDIOM_BLOCKS[idiom_id]
    return (
        f"## Idiomatic conventions for {label}\n\n"
        "An experienced developer on this stack would expect:\n\n"
        f"{bullets}\n\n"
    )
