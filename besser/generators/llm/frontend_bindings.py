"""Read literal props on generated JSX components without evaluating JavaScript.

Only JSON-valued expressions and quoted attributes are understood. Dynamic
bindings (including prop spreads) remain unknown, never guessed from a nested
lookup column's entity. This is deliberately not a general-purpose JSX parser.
"""
import json
import re


_COMPONENT = re.compile(r"<(TableBlock|MethodButton)\b")
_ATTRIBUTE = re.compile(r"[\w:-]+")
# Ignore JSX-looking examples in comments/string literals. Actual props are
# parsed separately below; this scan never evaluates interpolation or code.
# Only a template literal may span lines, so the two quoted alternatives stop
# at a newline: without that bound, two ordinary apostrophes in JSX prose
# ("Guest's" on one line, "Don't" on another) read as one string literal and
# swallowed the whole <MethodButton> between them, hiding a real blocker.
_NON_CODE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/"
    r"|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`"
)


def _expression_end(text: str, start: int) -> int | None:
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def literal_component_props(content: str):
    """Yield (source offset, component name, literal props) for generated tags."""
    consumed = 0
    non_code = iter(_NON_CODE.finditer(content))
    excluded = next(non_code, None)
    for match in _COMPONENT.finditer(content):
        if match.start() < consumed:
            continue
        while excluded and excluded.end() <= match.start():
            excluded = next(non_code, None)
        if excluded and excluded.start() <= match.start() < excluded.end():
            continue
        cursor = match.end()
        props = {}
        while cursor < len(content):
            if content[cursor].isspace():
                cursor += 1
                continue
            if content.startswith(("/>", ">"), cursor):
                consumed = cursor + (2 if content[cursor] == "/" else 1)
                yield match.start(), match.group(1), props
                break
            attribute = _ATTRIBUTE.match(content, cursor)
            if not attribute:  # Spreads or unsupported syntax: binding is unknown.
                break
            name = attribute.group()
            cursor = attribute.end()
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if cursor >= len(content) or content[cursor] != "=":
                continue  # Boolean prop.
            cursor += 1
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if cursor >= len(content):
                break
            if content[cursor] in "\"'":
                end = content.find(content[cursor], cursor + 1)
                if end < 0:
                    break
                props[name] = content[cursor + 1:end]
                cursor = end + 1
            elif content[cursor] == "{":
                end = _expression_end(content, cursor)
                if end is None:
                    break
                try:
                    props[name] = json.loads(content[cursor + 1:end - 1])
                except (ValueError, TypeError):
                    props[name] = None
                cursor = end
            else:
                break
