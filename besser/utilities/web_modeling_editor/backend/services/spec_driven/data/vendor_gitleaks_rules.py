"""Regenerate ``gitleaks_rules.toml`` from an upstream gitleaks release.

    python vendor_gitleaks_rules.py <gitleaks.toml> <version> > gitleaks_rules.toml

``<gitleaks.toml>`` is ``config/gitleaks.toml`` at the release tag
(https://github.com/gitleaks/gitleaks, MIT). Only prefix-anchored rules are
kept: the redacted secret must start with a literal of at least
``MIN_PREFIX`` characters (``ghp_``, ``AKIA``, ``-----BEGIN``). Keyword-
proximity and entropy-only rules (``generic-api-key``) are dropped, because
redaction rewrites source code and a false hit breaks the app.

The few Go RE2 constructs Python's ``re`` lacks are rewritten: a flag group
``(?i)`` after the start becomes a scoped ``(?i:...)``, ``\\z`` becomes
``\\Z``, ``\\Q...\\E`` is escaped and ``[:alnum:]`` is spelled out. A rule that
still does not compile is dropped and reported on stderr.

The leading ``[\\w.-]{0,50}?`` of a keyword rule (the identifier before
``mailgun``) is dropped: redaction replaces only the secret, and under
Python's backtracking ``re`` that prefix costs ~0.45 s per MB per rule, or
~17 s per MB where upstream nests it twice (``okta``, ``sumologic``).

A rule whose redacted span would be a constant is repaired (see
``without_constant_secret``) or dropped: redacting only the constant would
leave the token itself in place.
"""

import hashlib
import json
import re
import sys
import tomllib
import warnings
from re import _constants as sre
from re import _parser as sre_parse

MIN_PREFIX = 2
# Appended to a rule that is only a marker literal (``bedrock-api-key-YmVk...``):
# redacting the marker alone would leave the signed body in place.
MARKER_BODY = "[A-Za-z0-9+/=%&._-]*"
_POSIX_CLASSES = {"[:alnum:]": "a-zA-Z0-9", "[:alpha:]": "a-zA-Z", "[:digit:]": "0-9",
                  "[:lower:]": "a-z", "[:upper:]": "A-Z", "[:xdigit:]": "0-9A-Fa-f"}
_LICENSE = """\
MIT License

Copyright (c) 2019 Zachary Rice

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def to_python_regex(pattern: str) -> str:
    """``pattern`` (Go RE2) rewritten for Python's ``re``; ``ValueError`` when it cannot be."""
    pattern = re.sub(r"\\Q(.*?)\\E", lambda m: re.escape(m.group(1)), pattern)
    for posix, spelled in _POSIX_CLASSES.items():
        pattern = pattern.replace(posix, spelled)
    out, closers, i, in_class = [], [0], 0, False  # closers: scoped groups to close per paren depth
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            out.append("\\Z" if pattern[i + 1] == "z" else pattern[i:i + 2])
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
            if pattern[i + 1:i + 2] == "]":  # a leading ] is a literal
                out.append("[]")
                i += 2
                continue
        elif ch == "(":
            flags = re.match(r"\(\?([a-z]+)\)", pattern[i:])
            if flags and i > 0:
                out.append(f"(?{flags.group(1)}:")
                closers[-1] += 1
                i += flags.end()
                continue
            closers.append(0)
        elif ch == ")":
            out.append(")" * closers.pop())
        elif ch == "|" and closers[-1]:
            raise ValueError("a flag group before | changes the alternation")
        out.append(ch)
        i += 1
    out.append(")" * closers.pop())
    return "".join(out)


_IDENTIFIER_CONTEXT = re.compile(r"^(\(\?i\)|\(\?i:)?\[\\w\.-\]\{0,50\}\?")


def without_identifier_context(pattern: str) -> str:
    """``pattern`` without its leading ``[\\w.-]{0,50}?`` (repeated: upstream nests it)."""
    while match := _IDENTIFIER_CONTEXT.match(pattern):
        pattern = (match.group(1) or "") + pattern[match.end():]
    return pattern


def _compiles(pattern: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        re.compile(pattern, re.ASCII)


def _literal_prefixes(items) -> tuple[list[str], bool]:
    """The literal strings a parsed pattern can start with, and whether it is all literal."""
    prefixes = [""]
    for op, arg in items:
        if op is sre.AT:
            continue
        if op is sre.LITERAL:
            prefixes = [p + chr(arg) for p in prefixes]
        elif op is sre.IN and all(kind is sre.LITERAL for kind, _ in arg) and len(arg) <= 4:
            prefixes = [p + chr(c) for p in prefixes for _, c in arg]
        elif op is sre.SUBPATTERN or op is sre.BRANCH:
            branches = [arg[3]] if op is sre.SUBPATTERN else arg[1]
            results = [_literal_prefixes(list(b)) for b in branches]
            prefixes = [p + s for p in prefixes for subs, _ in results for s in subs]
            if not all(done for _, done in results):
                return prefixes, False
        elif op in (sre.MAX_REPEAT, sre.MIN_REPEAT) and arg[0] == 0 and _literal_prefixes(list(arg[2]))[1]:
            continue  # an optional literal, e.g. (?:https?://)?
        elif op in (sre.MAX_REPEAT, sre.MIN_REPEAT) and arg[0] == arg[1] and _literal_prefixes(list(arg[2]))[1]:
            subs = _literal_prefixes(list(arg[2]))[0]
            prefixes = [p + s * arg[0] for p in prefixes for s in subs]
        else:
            return prefixes, False
    return prefixes, True


def _group(items, number):
    for op, arg in items:
        if op is sre.SUBPATTERN and arg[0] == number:
            return list(arg[3])
        children = ([arg[3]] if op is sre.SUBPATTERN else arg[1] if op is sre.BRANCH
                    else [arg[2]] if op in (sre.MAX_REPEAT, sre.MIN_REPEAT) else [])
        for child in children:
            found = _group(list(child), number)
            if found is not None:
                return found
    return None


def _outer_groups(items) -> list[int]:
    """Capture groups not nested in another capture: the ones gitleaks' "first
    non-empty group" can pick, since an enclosing group wins over its inner ones."""
    found = []
    for op, arg in items:
        if op is sre.SUBPATTERN and arg[0] is not None:
            found.append(arg[0])
            continue
        children = ([arg[3]] if op is sre.SUBPATTERN else arg[1] if op is sre.BRANCH
                    else [arg[2]] if op in (sre.MAX_REPEAT, sre.MIN_REPEAT) else [])
        for child in children:
            found += _outer_groups(list(child))
    return found


def secret_shape(pattern: str, secret_group: int) -> tuple[int, bool]:
    """Shortest literal prefix of every span the rule could redact, and
    whether every such span is a constant (all literal)."""
    tree = sre_parse.parse(pattern, re.ASCII)
    groups = [secret_group] if secret_group else _outer_groups(list(tree)) or [0]
    spans = [list(tree) if g == 0 else _group(list(tree), g) for g in groups]
    shapes = [_literal_prefixes(s) for s in spans]
    return min(min(len(p) for p in prefixes) for prefixes, _ in shapes), all(done for _, done in shapes)


def without_constant_secret(pattern: str, secret_group: int) -> str:
    """``pattern`` changed so that what it redacts is not a constant.

    A marker literal (``bedrock-api-key-YmVk...``) gets ``MARKER_BODY``; named
    groups that only label a constant (``jwt-base64``'s ``(?P<alg>...)``) become
    non-capturing, so the whole match is the secret. Otherwise ``ValueError``.
    """
    if not secret_shape(pattern, secret_group)[1]:
        return pattern
    if not re.compile(pattern).groups:
        return pattern + MARKER_BODY
    unlabelled = re.sub(r"\(\?P<\w+>", "(?:", pattern)
    if secret_group or re.compile(unlabelled).groups:
        raise ValueError("the secret span is a constant")
    return unlabelled


def _toml_regexes(key: str, patterns: list[str]) -> str:
    return f"{key} = [\n" + "".join(f"    '''{p}''',\n" for p in patterns) + "]\n"


def _translated_allowlist(allowlist: dict, rule_id: str) -> "dict | None":
    """The allowlist with Python regexes, or None when it cannot apply here.

    A ``paths`` condition never applies: redaction sees text, not file paths.
    """
    if allowlist.get("paths") and allowlist.get("condition") == "AND":
        return None
    if not allowlist.get("regexes") and not allowlist.get("stopwords"):
        return None
    regexes = []
    for pattern in allowlist.get("regexes", []):
        try:
            regexes.append(to_python_regex(pattern))
            _compiles(regexes[-1])
        except (ValueError, re.error, Warning) as exc:
            raise ValueError(f"{rule_id}: allowlist regex {pattern!r}: {exc}") from None
    return {k: v for k, v in (("condition", allowlist.get("condition")),
                              ("regexTarget", allowlist.get("regexTarget")),
                              ("regexes", regexes), ("stopwords", allowlist.get("stopwords")))
            if v}


def _allowlist_toml(header: str, allowlist: dict) -> str:
    text = header
    for key in ("condition", "regexTarget"):
        if key in allowlist:
            text += f"{key} = {json.dumps(allowlist[key])}\n"
    if allowlist.get("regexes"):
        text += _toml_regexes("regexes", allowlist["regexes"])
    if allowlist.get("stopwords"):
        text += f"stopwords = {json.dumps(allowlist['stopwords'])}\n"
    return text


def main(source: str, version: str) -> None:
    raw = open(source, "rb").read()
    config = tomllib.loads(raw.decode("utf-8"))
    kept, dropped, body = 0, [], ""
    for rule in config["rules"]:
        rule_id = rule["id"]
        try:
            if "regex" not in rule:
                raise ValueError("path-only rule")
            pattern = without_identifier_context(to_python_regex(rule["regex"]))
            _compiles(pattern)
            pattern = without_constant_secret(pattern, rule.get("secretGroup", 0))
            prefix = secret_shape(pattern, rule.get("secretGroup", 0))[0]
            if prefix < MIN_PREFIX:
                raise ValueError(f"no literal prefix ({prefix} chars)")
            allowlists = [a for a in (_translated_allowlist(a, rule_id) for a in rule.get("allowlists", [])) if a]
        except (ValueError, re.error, Warning) as exc:
            dropped.append(f"{rule_id}: {exc}")
            continue
        kept += 1
        body += f"\n[[rules]]\nid = {json.dumps(rule_id)}\nregex = '''{pattern}'''\n"
        for key in ("secretGroup", "entropy"):
            if key in rule:
                body += f"{key} = {rule[key]}\n"
        body += f"keywords = {json.dumps(rule.get('keywords', []))}\n"
        for allowlist in allowlists:
            body += _allowlist_toml("[[rules.allowlists]]\n", allowlist)
    global_allowlist = _translated_allowlist({**config["allowlist"], "paths": None}, "global")

    license_text = "\n".join(f"# {line}".rstrip() for line in _LICENSE.splitlines())
    sys.stdout.write(
        f"# Prefix-anchored credential rules from gitleaks {version}\n"
        f"# (config/gitleaks.toml, sha256 {hashlib.sha256(raw).hexdigest()}).\n"
        "# Generated by vendor_gitleaks_rules.py; edit that script, not this file.\n"
        f"# {kept} of {len(config['rules'])} upstream rules kept; regexes rewritten for Python re.\n#\n"
        f"{license_text}\n\n"
        f"version = {json.dumps(version)}\n\n"
        + _allowlist_toml("[allowlist]\n", global_allowlist) + body
    )
    print(f"kept {kept}, dropped {len(dropped)}:", *dropped, sep="\n  ", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
