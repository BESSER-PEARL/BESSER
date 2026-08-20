#!/usr/bin/env bash
#
# Build the Sphinx docs and FAIL on any new / non-allowlisted warning.
#
# Why a gate: the docs had accumulated hundreds of warnings, which meant a real
# regression (a broken cross-reference, a missing page, a bad heading underline,
# an unknown code-block lexer, a malformed docstring) would hide in the noise.
# This gate builds the docs and fails the build unless every warning belongs to
# one of the two pre-existing *systemic* categories below — so new drift can no
# longer land silently, while the known structural noise doesn't block CI.
#
# Allowlisted (pre-existing, structural — tracked separately, not regressions):
#   * "duplicate object description"
#       autodoc documents the same objects twice because packages re-export
#       their submodule's classes (``from .x import *``); the package page and
#       the submodule page both index them. Fixing this means restructuring the
#       API-reference autodoc across the metamodel; out of scope for this gate.
#   * "more than one target found for cross-reference"
#       ambiguous class names shared across modules (e.g. Transition / Event /
#       Parameter exist in both the state-machine/gui and the ocl metamodels).
#
# Anything else -> non-zero exit. Run locally the same way CI does:
#   pip install -e . && pip install -r docs/requirements.txt && bash docs/check-docs-warnings.sh
#
set -uo pipefail
cd "$(dirname "$0")"

BUILD_LOG="$(mktemp)"
trap 'rm -f "$BUILD_LOG"' EXIT

# -E: don't use a saved environment (fresh build so every warning re-emits).
# --keep-going: report all problems, don't stop at the first.
# -w: also capture warnings to a file we can filter.
python -m sphinx -b html -E --keep-going -w "$BUILD_LOG" source build/html >/dev/null 2>&1 || true

ALLOWLIST='duplicate object description|more than one target found for cross-reference'

UNEXPECTED="$(grep -iE 'WARNING|ERROR' "$BUILD_LOG" 2>/dev/null | grep -viE "$ALLOWLIST" || true)"

if [ -n "$UNEXPECTED" ]; then
  echo "::error::Sphinx produced non-allowlisted documentation warnings/errors:"
  echo "-------------------------------------------------------------------------"
  echo "$UNEXPECTED"
  echo "-------------------------------------------------------------------------"
  echo "Fix them, or (only if genuinely a new structural category) extend the"
  echo "ALLOWLIST in docs/check-docs-warnings.sh with justification."
  exit 1
fi

echo "Docs build clean — only the two allowlisted systemic warning categories are present."
