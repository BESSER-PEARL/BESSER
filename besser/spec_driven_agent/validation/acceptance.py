"""Model-derived acceptance matrix for generated apps.

For every class in the domain model, three static discovery signals
indicate whether the entity appears in the app (not executed acceptance):

* ``route``  — the backend exposes REST routes for it (checked against
  the same static route parse the endpoint manifest uses). That parse
  reads FastAPI decorators in ``.py`` files only, so on any other stack
  the cell is ``None`` (unmeasured) rather than ``False``;
* ``page``   — some frontend file is about it (name or content match);
* ``create`` — a frontend file about it contains a POST, OR calls a
  shared api-client helper or React hook that issues one (a generic form is about the
  entities its imported config module declares), OR a ``<TableBlock>`` bound
  to it: ``dataBinding`` naming this entity and an endpoint, plus a
  non-empty ``options.formColumns``. The GUI-model scaffold's only
  generated create path issues its POST from a shared runtime component
  (``TableComponent``), never a literal call in the page file itself, so
  the binding — not the call site — is the signal. An abstract class is
  created through its subclasses, so its cell is ``None`` (not expected);
  so is the cell of a class a GUI model's screens offer no way to create,
  when the screens bind at least one class.

Most generated React frontends put every request behind one api module
(``api.create('carpark', payload)`` → ``fetch(url, {method:'POST'})``),
so the literal POST never appears in the file that names the entity, and a
``.post(`` scan alone would report "no create path" for working apps.
``_client_create_calls`` resolves those calls instead of demanding a
literal in one file.

The matrix is deliberately REPORT-ONLY (warnings + a recipe field, never
blockers): a GUI-model-driven run may legitimately scope the UI to a
subset of entities, so "entity X has no page" is a visibility signal for
the checklist and the recipe, not a defect the fix loop must burn turns
on. Blocker-level enforcement stays with ``contract_checks``.
"""

from __future__ import annotations

import bisect
import functools
import os
import re
from besser.BUML.metamodel.gui import Button, ButtonActionType, Form
from besser.spec_driven_agent.validation.constructibility import abstract_entities
from besser.spec_driven_agent.validation.endpoint_coherence import _GENERATED_CLIENT_RE
from besser.spec_driven_agent.validation.frontend_forms import _attr, _jsx_tags
from besser.utilities.workspace_fs import walk_plain

_FRONTEND_EXTS = (".js", ".jsx", ".ts", ".tsx")
_SKIP_DIRS = ("node_modules", "dist", "build", "__pycache__")

# A POST issued from frontend code — axios/api `.post(`, the method as an
# argument (`request('POST', path)`), or fetch with method POST; `url` is the
# literal path a fetch-style call names before its options.
_POST_RE = re.compile(
    r"\.post\s*\(|\(\s*['\"]POST['\"]\s*,"
    r"|(?:\(\s*['\"](?P<url>[^'\"]{1,200})['\"]\s*,\s*\{[^{}]{0,200}?)?method\s*:\s*['\"]POST['\"]",
    re.IGNORECASE)
_LITERAL_PATH_RE = re.compile(r"\s*['\"]([^'\"]{1,200})['\"]\s*")
_TEMPLATE_SLOT_RE = re.compile(r"\$\{[^}]*\}")
# A one-expression arrow body up to the call that POSTs: `=> api`, `=> request<T>`.
_CALL_HEAD_RE = re.compile(r"[\w$.\s]*(?:<[^<>()]*>)?\s*")
_BRACKET_RE = re.compile(r"[()\[\]{}]")
# A helper handed over as a value sits after one of these and before `_REFERENCE_AFTER_RE`.
_REFERENCE_BEFORE = frozenset(":=({,[")
_REFERENCE_AFTER_RE = re.compile(r"\s*(?:[,)}\];]|\Z)")
# Characters (a bracket group counts as one) a type annotation or type
# parameter list is read for: real ones are short, and an unbounded scan from
# every candidate name is quadratic on a long line with no terminator.
_MAX_TYPE_SCAN = 200
# What a name with a type annotation follows: a declaration keyword, or the
# start of a line (a class field). Not `{`/`,` (an object key) nor `?` (a ternary).
_DECLARATION_RE = re.compile(
    r"(?:\b(?:const|let|var|readonly|static|public|private|protected|declare|export)\s+|\n[ \t]*)\Z")
_DECLARATION_WINDOW = 40

# The GUI-model scaffold's runtime-bound create path: a <TableBlock> whose
# dataBinding names the entity + endpoint and whose options carry at least
# one editable formColumns entry. Its POST is issued by the shared
# TableComponent it delegates to (axios.post gated on modalMode === 'add'),
# not by a literal call in the page file — a `.post(` scan is structurally
# blind to it. DOTALL: the scaffold sometimes pretty-prints this tag across
# many lines; dataBinding's own body never contains `}` so `[^}]*` closes on
# the real end without needing DOTALL there.
_TABLE_BLOCK_RE = re.compile(r"<TableBlock\b.*?/>", re.DOTALL)
_DATA_BINDING_RE = re.compile(r"dataBinding\s*=\s*\{\{([^}]*)\}\}")
_ENTITY_RE = re.compile(r'"entity"\s*:\s*"([^"]*)"')
_ENDPOINT_RE = re.compile(r'"endpoint"\s*:\s*"([^"]*)"')
_NONEMPTY_FORM_COLUMNS_RE = re.compile(r'"formColumns"\s*:\s*\[\s*[^\]\s]')

# Resolution of a create issued through a shared api-client module.
_IMPORT_RE = re.compile(
    r"\bimport\s+(?P<clause>[^'\";]+?)\s+from\s+['\"](?P<spec>\.[^'\"]*)['\"]")
_NAMESPACE_IMPORT_RE = re.compile(r"\*\s+as\s+([A-Za-z_$][\w$]*)")
_NAMED_IMPORT_RE = re.compile(r"\{([^}]*)\}")
_LEADING_NAME_RE = re.compile(r"^([A-Za-z_$][\w$]*)")
# A name a callable definition can start with: `name(`, `name<T>(`, `name =`,
# `name:`. ``_definition_body`` decides whether a function follows; a value
# that is not one (`const response = await fetch(...)`) must not read as the
# helper enclosing the POST, or the real one around it is never found.
_DEF_NAME_RE = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*(?=[(<:=])")
_WORD_RE = re.compile(r"[A-Za-z_$][\w$]*")
_RETURN_TYPE_RE = re.compile(r"\s*:")
_NOT_A_HELPER = frozenset({
    "if", "for", "while", "switch", "catch", "function", "return", "await",
    "typeof", "new", "delete", "void", "do", "else", "case", "default",
})
_STRING_ARG_RE = re.compile(r"['\"]([^'\"]{1,60})['\"]")
_BLANKABLE_RE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/"
    r"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`")
_KEEP_NEWLINES_RE = re.compile(r"[^\n]")
# A string compared against a value (`key === 'booking'`, `case 'booking':`).
_COMPARED_STRING_RE = re.compile(
    r"(?:[!=]==?|\bcase)\s*(['\"`])[^'\"`\n]*\1|(['\"`])[^'\"`\n]*\2\s*[!=]==?")
# An identifier holding another record's id (`tagId`, `user_ids`): a reference, not a create.
_REFERENCE_ID_RE = re.compile(r"(?<![\w$])[A-Za-z_$][\w$]*?(?:Id|ID|_id)s?(?![\w$])")
_IF_RE = re.compile(r"\bif\s*\(")
_BLOCK_START_RE = re.compile(r"\s*\{")
# An error or response read (`err.message`, `e?.response.data`, `error['message']`), not a call.
_ERROR_MEMBER = r"(?:message|error|detail|data|status|statusText|response|stack|code|name)"
_ERROR_READ_RE = re.compile(
    r"(?<![\w$.])(?:e|err|error|ex|exception|res|resp|response|result)"
    r"(?:\s*\??\.\s*" + _ERROR_MEMBER + r"(?![\w$])(?!\s*\()"
    r"|\s*(?:\?\.)?\[\s*(['\"`])" + _ERROR_MEMBER + r"\1\s*\])+")


def _blank_literals(text: str) -> str:
    """Blank comments and string bodies, keeping every offset in place.

    Offsets and newlines are preserved so `_HELPER_DEF_RE` positions still
    index into the original source. Substituting only the matched spans keeps
    memory proportional to the literals, not to the file.
    """
    return _BLANKABLE_RE.sub(lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group()), text)


def _blank_comments(text: str) -> str:
    """Blank comments only: a doc comment's example path names no resource."""
    return _BLANKABLE_RE.sub(
        lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group()) if m.group().startswith("/") else m.group(), text)


def _blank_prose(text: str) -> str:
    """Blank comments and every string holding a space: a message or a label, not a name."""
    return _BLANKABLE_RE.sub(
        lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group())
        if m.group().startswith("/") or re.search(r"\s", m.group()) else m.group(), text)


def _record_id_piece(expression: str) -> str | None:
    """The path text of a part computed at run time: nothing for a record id
    (`cartId`, `cart.id`), None for anything else, which may name the resource."""
    name = re.split(r"[^\w$]", expression.strip())[-1]
    return "" if name == "id" or name.endswith(("Id", "ID", "_id")) else None


def _argument_path(source: str, blanked: str, start: int) -> str | None:
    """The literal text of the call argument at ``start``, or None when it names no path.

    A plain literal is the path. ``API + '/booking/'`` and `` `${API}/booking/` ``
    keep their literal parts; when no letter is left (`` `/${route}/` ``) the
    path is computed. A record id in it (`` `/checkout/${cartId}/` ``) leaves the
    path literal. An options object (`axios({ url, method })`) is no path.
    """
    end, operands = start, [start]
    while end < len(blanked) and blanked[end] not in ",)]}":
        if blanked[end] == "+":
            operands.append(end + 1)
        end = _group_end(blanked, end) if blanked[end] in "([{" else end + 1
    text = source[start:end]
    leading = _LITERAL_PATH_RE.match(text)
    if leading:
        return leading.group(1)
    if text.lstrip().startswith("{"):
        return None
    # The literal parts, and None for each part computed at run time.
    pieces: list[str | None] = []
    for begin, stop in zip(operands, [o - 1 for o in operands[1:]] + [end]):
        operand = source[begin:stop].strip()
        literal = _BLANKABLE_RE.fullmatch(operand)
        if literal and operand[0] in "'\"":
            pieces.append(operand[1:-1])
        elif literal and operand[0] == "`":
            body, at = operand[1:-1], 0
            for slot in _TEMPLATE_SLOT_RE.finditer(body):
                pieces += [body[at:slot.start()], _record_id_piece(slot.group()[2:-1])]
                at = slot.end()
            pieces.append(body[at:])
        else:
            pieces.append(_record_id_piece(operand))
    while pieces and not pieces[0]:
        pieces.pop(0)  # the base URL
    # A slot after the base that is no record id names the resource (`/api/${resource}/`): computed.
    if None in pieces:
        return None
    static = "".join(pieces)
    return static if re.search(r"[A-Za-z]", static) else None


def _options_call_start(blanked: str, method_at: int) -> int | None:
    """Where the first argument starts of the call whose options object holds
    the ``method:`` at ``method_at`` (`fetch(url, { method: 'POST' })`)."""
    options = _innermost_open(blanked, method_at)
    call = _innermost_open(blanked, options) if options >= 0 else -1
    if options < 0 or call < 0 or blanked[options] != "{" or blanked[call] != "(":
        return None
    return call + 1


def _post_path(source: str, blanked: str, post: re.Match) -> str | None:
    """The path a POST targets (its literal parts), or None when it is built at run time."""
    if post.group("url"):
        return post.group("url")
    if post.group().endswith(("(", ",")):
        return _argument_path(source, blanked, post.end())
    start = _options_call_start(blanked, post.start())
    return None if start is None else _argument_path(source, blanked, start)


def _direct_create_calls(source: str) -> list:
    """Creates ``source`` issues itself, shaped like ``_client_create_calls``."""
    code = _blank_comments(source)
    blanked = _blank_literals(code)
    return [[path] if (path := _post_path(code, blanked, m)) else None for m in _POST_RE.finditer(code)]


_BracketTable = tuple[list[int], list[int], list[int], dict[int, int]]
# The text last asked about and its table. Looked up by identity: an equality
# lookup compares two copies of a large file on every call. Read once into a
# local, so a thread that swaps it in between never mixes tables.
_last_brackets: tuple[str, _BracketTable] | None = None


def _brackets(blanked: str) -> _BracketTable:
    """``_bracket_table(blanked)``, built once per text."""
    global _last_brackets
    last = _last_brackets
    if last is None or last[0] is not blanked:
        last = _last_brackets = (blanked, _bracket_table(blanked))
    return last[1]


def _bracket_table(blanked: str) -> _BracketTable:
    """One pass over the brackets of ``blanked``: their positions, the depth and
    the innermost open bracket (-1 for none) after each, and the index past the
    group each opener starts. Every kind counts alike; an opener never closed
    runs to the end of the text."""
    positions: list[int] = []
    depths: list[int] = []
    tops: list[int] = []
    group_end: dict[int, int] = {}
    stack: list[int] = []
    for match in _BRACKET_RE.finditer(blanked):
        i = match.start()
        if blanked[i] in "([{":
            stack.append(i)
        elif stack:
            group_end[stack.pop()] = i + 1
        positions.append(i)
        depths.append(len(stack))
        tops.append(stack[-1] if stack else -1)
    return positions, depths, tops, group_end


def _depth_before(blanked: str, i: int) -> int:
    positions, depths, _, _ = _brackets(blanked)
    k = bisect.bisect_left(positions, i)
    return depths[k - 1] if k else 0


def _innermost_open(blanked: str, i: int) -> int:
    """The innermost bracket still open just before ``i``, or -1."""
    positions, _, tops, _ = _brackets(blanked)
    k = bisect.bisect_left(positions, i)
    return tops[k - 1] if k else -1


def _bracket_depth(blanked: str, start: int, end: int) -> int:
    """The net bracket depth of ``blanked[start:end]``."""
    return _depth_before(blanked, end) - _depth_before(blanked, start)


def _group_end(blanked: str, start: int) -> int:
    """The index just past the bracket group that opens at ``start``."""
    return _brackets(blanked)[3].get(start, len(blanked))


def _blank_entity_branches(code: str, forms: list[str]) -> str:
    """Blank each `if (key === 'project') { ... }`: a shared form's branch for one entity it serves."""
    blanked, done = _blank_literals(code), 0
    for match in _IF_RE.finditer(blanked):
        cond_end = _group_end(blanked, match.end() - 1)
        body = _BLOCK_START_RE.match(blanked, cond_end)
        if match.start() < done or not body or not any(
                _mentions(compared.group(), forms, whole_word=True)
                for compared in _COMPARED_STRING_RE.finditer(code, match.start(), cond_end)):
            continue
        done = _group_end(blanked, body.end() - 1)
        code = code[:match.start()] + _KEEP_NEWLINES_RE.sub(" ", code[match.start():done]) + code[done:]
    return code


def _names_an_entity(code: str, forms: list[str]) -> bool:
    """Whether a component's code names an entity beyond what a shared form holds
    for any of them: a test of the entity it was given and the block it runs for
    it, an error or response read, another record's id."""
    return _mentions(_REFERENCE_ID_RE.sub(" ", _ERROR_READ_RE.sub(" ", _COMPARED_STRING_RE.sub(
        " ", _blank_entity_branches(code, forms)))), forms)


def _skip_space(blanked: str, i: int) -> int:
    while i < len(blanked) and blanked[i].isspace():
        i += 1
    return i


def _angle_end(blanked: str, i: int) -> int | None:
    """Past the ``<...>`` type parameters opening at ``i``, or None when it is not a list of them."""
    depth = 0
    for _ in range(_MAX_TYPE_SCAN):
        if i >= len(blanked):
            return None
        char = blanked[i]
        if char in "([{":
            i = _group_end(blanked, i)
            continue
        if char in ";)]}":
            return None
        if char == "<":
            depth += 1
        elif char == ">" and blanked[i - 1] != "=":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _params_body(blanked: str, i: int, arrow: bool) -> int | None:
    """Where the body starts of ``[<T>](params)[: R]`` at ``i``, followed by ``{``
    or, for an arrow function, ``=>``; None when that is not what follows."""
    if blanked.startswith("<", i):
        end = _angle_end(blanked, i)
        if end is None:
            return None
        i = _skip_space(blanked, end)
    if not blanked.startswith("(", i):
        return None
    i = _skip_space(blanked, _return_type_end(blanked, _group_end(blanked, i)))
    if arrow:
        return _skip_space(blanked, i + 2) if blanked.startswith("=>", i) else None
    return i if blanked.startswith("{", i) else None


def _function_value_body(blanked: str, i: int) -> int | None:
    """Where the body starts of a function expression or arrow function at ``i``."""
    word = _WORD_RE.match(blanked, i)
    if word and word.group() == "async":
        i = _skip_space(blanked, word.end())
        word = _WORD_RE.match(blanked, i)
    if word and word.group() == "function":
        i = _skip_space(blanked, word.end())
        i = _skip_space(blanked, i + 1) if blanked.startswith("*", i) else i
        name = _WORD_RE.match(blanked, i)
        return _params_body(blanked, _skip_space(blanked, name.end()) if name else i, arrow=False)
    if word:  # one parameter, no parentheses: `d => ...`
        i = _skip_space(blanked, word.end())
        return _skip_space(blanked, i + 2) if blanked.startswith("=>", i) else None
    return _params_body(blanked, i, arrow=True)


def _annotation_end(blanked: str, i: int) -> int | None:
    """The ``=`` that ends the ``: Type`` annotation starting at ``i``, or None."""
    angle = 0
    for _ in range(_MAX_TYPE_SCAN):
        if i >= len(blanked):
            return None
        char = blanked[i]
        if char in "([{":
            i = _group_end(blanked, i)
            continue
        if char in ";)]}\n" or char == "," and angle <= 0:
            return None
        if char == "=":
            if blanked.startswith("=>", i):
                i += 2
                continue
            return None if blanked.startswith("==", i) else i
        angle += (char == "<") - (char == ">")
        i += 1
    return None


def _definition_body(blanked: str, i: int, typed: bool) -> int | None:
    """Where the body starts of the callable whose name ends at ``i``: a function
    or method, a function bound by ``=`` or ``:``, or one bound with a type
    (`const save: Handler = async () => {`). None when the name defines no body
    (a call, a plain value, ``declare function f(): T;``). ``typed``: the name
    is a declaration (`const x`, a class field), so ``name: Type =`` may follow."""
    i = _skip_space(blanked, i)
    if blanked.startswith("=", i):
        if blanked.startswith(("==", "=>"), i):
            return None
        return _function_value_body(blanked, _skip_space(blanked, i + 1))
    if blanked.startswith(":", i):
        value = _skip_space(blanked, i + 1)
        body = _function_value_body(blanked, value)
        equals = _annotation_end(blanked, value) if body is None and typed else None
        return body if equals is None else _function_value_body(blanked, _skip_space(blanked, equals + 1))
    return _params_body(blanked, i, arrow=False)


def _callable_definitions(blanked: str) -> list[tuple[int, str]]:
    """``(body start, name)`` of every named callable definition, in source order.

    A brace body starts at its ``{``; a one-expression arrow body at the expression.
    A name inside a definition's head (the ``Handler`` of `save: Handler = ...`,
    a parameter's default) is part of that definition, not another one.
    """
    definitions: list[tuple[int, str]] = []
    for match in _DEF_NAME_RE.finditer(blanked):
        if match.group(1) in _NOT_A_HELPER or definitions and match.start() < definitions[-1][0]:
            continue
        begin = max(0, match.start() - _DECLARATION_WINDOW)
        window = ("" if begin else "\n") + blanked[begin:match.start()]  # the file start is a line start
        body = _definition_body(blanked, match.end(), typed=bool(_DECLARATION_RE.search(window)))
        if body is not None:
            definitions.append((body, match.group(1)))
    return definitions


def _post_helper_paths(source: str) -> dict[str, list[str] | None]:
    """{helper name: the literal paths its POSTs target, or None if any is not literal}.

    For each POST literal, the enclosing helper is the last named callable
    definition before it that we are still syntactically inside, or whose
    one-expression arrow body it is. Only those names count as a create;
    `list`/`update`/`remove` next to them do not.
    """
    blanked = _blank_literals(source)
    definitions = _callable_definitions(blanked)
    starts = [start for start, _ in definitions]  # ascending: see _callable_definitions
    helpers: dict[str, list[str] | None] = {}
    for post in _POST_RE.finditer(source):
        for k in range(bisect.bisect_right(starts, post.start()) - 1, -1, -1):
            start, name = definitions[k]
            if (_bracket_depth(blanked, start, post.start()) >= 1
                    or _CALL_HEAD_RE.fullmatch(blanked, start, post.start())):
                path, known = _post_path(source, blanked, post), helpers.get(name, [])
                helpers[name] = None if path is None or known is None else known + [path]
                break
    return helpers


def _body_end(blanked: str, start: int) -> int:
    """Where the body starting at ``start`` ends: its brace group, or for an
    arrow's expression the first bracket group in it."""
    positions = _brackets(blanked)[0]
    k = bisect.bisect_left(positions, start)
    if k == len(positions):
        return len(blanked)
    j = positions[k]
    return j if blanked[j] in ")]}" else _group_end(blanked, j)


def _return_type_end(blanked: str, i: int) -> int:
    """Past a ``: Type`` annotation at ``i``. A ``{`` right after ``:``, ``|``,
    ``&`` or ``,`` (or inside ``<>``) is an object type; any other ``{``, or a
    ``=>`` outside brackets, starts the body."""
    colon = _RETURN_TYPE_RE.match(blanked, i)
    if not colon:
        return i
    i, angle, previous = colon.end(), 0, ":"
    for _ in range(_MAX_TYPE_SCAN):
        if i >= len(blanked):
            break
        char = blanked[i]
        if char.isspace():
            i += 1
            continue
        if char in ";)]}" or angle <= 0 and (
                blanked.startswith("=>", i) or char == "{" and previous not in ":|&,"):
            return i
        if char in "([{":
            i, previous = _group_end(blanked, i), ")"
            continue
        if char == "<":
            angle += 1
        elif char == ">" and previous != "=":
            angle -= 1
        i, previous = i + 1, char
    return i


def _hook_post_paths(source: str) -> dict[str, list[str] | None]:
    """{React hook name: the literal paths of the POSTs inside its body, or None if any is not literal}.

    A page calling ``useBookings()`` creates through whatever the hook returns
    (``{ add }``), so the hook call is that page's create.
    """
    blanked = _blank_literals(source)
    posts = list(_POST_RE.finditer(source))
    post_starts = [post.start() for post in posts]
    hooks: dict[str, list[str] | None] = {}
    for start, name in _callable_definitions(blanked):
        if not re.fullmatch(r"use[A-Z][\w$]*", name):
            continue
        end = _body_end(blanked, start)
        for post in posts[bisect.bisect_left(post_starts, start):bisect.bisect_left(post_starts, end)]:
            path, known = _post_path(source, blanked, post), hooks.get(name, [])
            hooks[name] = None if path is None or known is None else known + [path]
    return hooks


def _post_helpers(source: str) -> set[str]:
    """Names in an api-client module whose own definition issues a POST."""
    return set(_post_helper_paths(source))


def _import_bindings(source: str):
    """Yield (local name, relative specifier) for each relative import."""
    for match in _IMPORT_RE.finditer(source):
        clause, spec = match.group("clause"), match.group("spec")
        namespace = _NAMESPACE_IMPORT_RE.search(clause)
        if namespace:
            yield namespace.group(1), spec
        named = _NAMED_IMPORT_RE.search(clause)
        if named:
            for item in named.group(1).split(","):
                item = item.strip()
                if item:
                    yield item.split(" as ")[-1].strip(), spec
        default = clause.split("{")[0].split("*")[0].strip().rstrip(",").strip()
        if default:
            leading = _LEADING_NAME_RE.match(default)
            if leading:
                yield leading.group(1), spec


def _resolve_import(rel: str, spec: str) -> list[str]:
    parts = rel.split("/")[:-1]
    for segment in spec.split("/"):
        if segment == "..":
            if parts:
                parts.pop()
        elif segment not in ("", "."):
            parts.append(segment)
    stem = "/".join(parts)
    candidates = [stem] if stem.endswith(_FRONTEND_EXTS) else []
    return (candidates + [stem + ext for ext in _FRONTEND_EXTS]
            + [f"{stem}/index{ext}" for ext in _FRONTEND_EXTS])


def _import_targets(rel: str, source: str, files: dict[str, str]) -> dict[str, str]:
    """{local name: the workspace file it is imported from} for the relative imports of ``rel``."""
    return {binding: target for binding, spec in _import_bindings(source)
            for target in [next((c for c in _resolve_import(rel, spec) if c in files), None)] if target}


def _previous_char(text: str, i: int) -> str:
    """The last non-space character before ``i``, or ``""``."""
    while i > 0 and text[i - 1].isspace():
        i -= 1
    return text[i - 1] if i else ""


# A call's first argument read off an object: `config.path`, `ENTITIES[key].path`.
_PROPERTY_ARG_RE = re.compile(
    r"\s*(?P<root>[A-Za-z_$][\w$]*)(?P<chain>(?:\s*\??\.\s*[\w$]+|\s*\[[^\[\]]*\])*?)"
    r"\s*\??\.\s*(?P<prop>[A-Za-z_$][\w$]*)\s*(?:,|\Z)")
# The entry of an object a local name is bound to: `const config = ENTITIES[entityKey]`.
_ENTRY_BINDING = r"\b(?:const|let|var)\s+{name}\s*=\s*(?P<base>[A-Za-z_$][\w$]*)\s*(?:\?\.)?\["
# Depth of an entry's own property in `const ENTITIES = { key: { path: 'x' } }` (or `[{ path: 'x' }]`).
_ENTRY_PROPERTY_DEPTH = 2
# A component's destructured props holding ``name``: `function Page({ entity, mode = 'list' })`.
_PROPS_PARAMETER = r"\(\s*\{{[^{{}}()]*(?<![\w$]){name}(?![\w$])[^{{}}()]*\}}"
# One entry of a metadata object, or any of them: `ENTITIES.order`, `ENTITIES['order']`, `ENTITIES[key]`.
_ENTRY_EXPRESSION_RE = re.compile(
    r"\s*(?P<base>[A-Za-z_$][\w$]*)\s*(?:\??\.\s*(?P<key>[A-Za-z_$][\w$]*)"
    r"|(?:\?\.)?\s*\[\s*(?:(['\"])(?P<quoted>[^'\"\[\]]+)\3|[^\[\]]*)\s*\])\s*")
# An entry that says its page offers no create form: `allowCreate: false`, `readOnly: true`.
_NO_CREATE_RE = re.compile(
    r"(?<![\w$.])(?:(?:allowCreate|canCreate|creatable)\s*:\s*false|(?:readOnly|readonly)\s*:\s*true)\b")
# The key an object entry is declared under, right before its `{`.
_ENTRY_KEY_BEFORE_RE = re.compile(r"(['\"]?)[\w$]+\1\s*:\s*$")


def _without_uncreatable_entries(source: str) -> str:
    """``source`` with every object entry that declares no create form blanked, key included."""
    blanked = _blank_literals(source)
    spans = set()
    for flag in _NO_CREATE_RE.finditer(blanked):
        opened = _innermost_open(blanked, flag.start())
        if opened < 0 or blanked[opened] != "{":
            continue
        key = _ENTRY_KEY_BEFORE_RE.search(source, 0, opened)
        spans.add((key.start() if key else opened, _group_end(blanked, opened)))
    for start, end in sorted(spans, reverse=True):
        source = source[:start] + _KEEP_NEWLINES_RE.sub(" ", source[start:end]) + source[end:]
    return source


def _leading_expression(argument: str) -> str:
    """The expression a create's path argument starts with: `record.path` in
    `` `${record.path}?owner=${id}` ``, in `record.path + 'bulk/'` and in `record.path`."""
    slot = re.match(r"\s*`\$\{([^{}]*)\}", argument)
    return slot.group(1) if slot else re.split(r"[+,]", argument, maxsplit=1)[0]


def _metadata_values(arg_text: str, code: str, bindings: dict[str, str], files: dict[str, str]) -> list[str]:
    """The literal values a create's first argument can take when it reads a
    property of an entry of an imported metadata object, else ``[]``.

    `api.create(config.path, ...)` with `const config = ENTITIES[entityKey]` and
    `ENTITIES` imported from `entities.js` takes the `path` of each entry of
    `ENTITIES` there. Only literal values of that property, at entry level of
    that object, count: an entity the object does not list gets nothing.
    """
    read = _PROPERTY_ARG_RE.match(arg_text)
    if read is None:
        return []
    base = read.group("root")
    if not read.group("chain").lstrip().startswith("["):
        entry = re.search(_ENTRY_BINDING.format(name=re.escape(base)), code)
        if entry is None or read.group("chain").strip():
            return []
        base = entry.group("base")
    return _entry_values(base, read.group("prop"), None, bindings, files)


def _entry_values(base: str, prop: str, key: str | None,
                  bindings: dict[str, str], files: dict[str, str]) -> list[str]:
    """The literal values of ``prop`` in the entries of the imported object ``base``,
    or in its entry ``key`` alone when one is given; ``[]`` when nothing declares them."""
    module = files.get(bindings.get(base, ""))
    if module is None:
        return []
    module = _without_uncreatable_entries(_blank_comments(module))
    blanked = _blank_literals(module)
    declared = re.search(r"\b(?:const|let|var)\s+" + re.escape(base) + r"\b[^=;]*=\s*(?=[{\[])", blanked)
    if declared is None:
        return []
    start, depth = declared.end(), _ENTRY_PROPERTY_DEPTH
    if key is not None:
        named = re.compile(r"(?<![\w$.])(['\"]?)" + re.escape(key) + r"\1\s*:\s*(?=\{)")
        entry = next((m for m in named.finditer(module, start, _group_end(blanked, start))
                      if _bracket_depth(blanked, start, m.start()) == 1), None)
        if entry is None:
            return []
        start, depth = entry.end(), 1
    end = _group_end(blanked, start)
    values = re.compile(r"(?<![\w$.])" + re.escape(prop) + r"\s*:\s*(['\"])([^'\"\n]{1,100})\1")
    return [value.group(2) for value in values.finditer(module, start, end)
            if _bracket_depth(blanked, start, value.start()) == depth]


def _prop_read(arg_text: str, code: str) -> tuple[str, str] | None:
    """``(prop, property)`` when a create's first argument reads a property of a
    component prop (`entity.path` in `function Page({ entity })`), else None."""
    read = _PROPERTY_ARG_RE.match(arg_text)
    if read is None or read.group("chain").strip():
        return None
    name = read.group("root")
    if not re.search(_PROPS_PARAMETER.format(name=re.escape(name)), code):
        return None
    return name, read.group("prop")


# The JSX attributes, and the value of each, that switch a component's create form off.
_NO_CREATE_ATTRS = (("readOnly", "true"), ("readonly", "true"),
                    ("allowCreate", "false"), ("canCreate", "false"), ("creatable", "false"))


def _turns_create_off(attrs: str, top: str) -> bool:
    """Whether a JSX tag switches its create form off: `readOnly`, `readOnly={true}`, `allowCreate={false}`."""
    return (re.search(r"(?<![\w$.-])(?:readOnly|readonly)(?![\w$-])(?!\s*=)", top) is not None
            or any((_attr(attrs, name) or "").strip() == value for name, value in _NO_CREATE_ATTRS))


def _forwarded_props(source: str, binding: str, prop: str) -> list[str]:
    """The component's own props ``source`` hands on to ``binding`` as ``prop``:
    `<EntityForm entity={entity} />` in `function EntityPage({ entity })`."""
    code = _blank_literals(source)
    names = []
    for _, attrs, top in _jsx_tags(_blank_comments(source), re.escape(binding)):
        value = (_attr(attrs, re.escape(prop)) or "").strip()
        if (not _turns_create_off(attrs, top) and re.fullmatch(r"[A-Za-z_$][\w$]*", value)
                and re.search(_PROPS_PARAMETER.format(name=re.escape(value)), code)):
            names.append(value)
    return names


def _rendered_entry_values(source: str, binding: str, prop: str, entry_property: str,
                           bindings: dict[str, str], files: dict[str, str]) -> list[str]:
    """The literal values of ``entry_property`` in the metadata entries ``source``
    hands to the component ``binding`` as ``prop``: `<EntityPage entity={ENTITIES.order} />`."""
    values: list[str] = []
    for _, attrs, top in _jsx_tags(_blank_comments(source), re.escape(binding)):
        if _turns_create_off(attrs, top):
            continue
        entry = _ENTRY_EXPRESSION_RE.fullmatch(_attr(attrs, re.escape(prop)) or "")
        if entry:
            key = entry["key"] or entry["quoted"]
            values += _entry_values(entry["base"], entry_property, key, bindings, files)
    return values


def _client_create_calls(rel: str, source: str, files: dict[str, str],
                         helper_cache: dict[str, dict], used_clients: set[str],
                         traced: list[str] | None = None,
                         prop_reads: list[tuple[str, str]] | None = None,
                         handed: list[str] | None = None) -> list:
    """Creates ``rel`` issues through an imported api-client module.

    One entry per call site: the helper's own literal path, else the call's
    literal string arguments, or None when the call names its resource
    through a variable (the generic ``<EntityList entity={...}/>`` page).
    None means "a create happens here but this call does not say which
    entity", which is exactly the evidence the file-mention test already
    stands on. Each module a call resolves into is added to ``used_clients``.
    A helper handed over uncalled (`mutationFn: api.createBooking`) is a call
    site too, and so is a method called on an instance of an imported class.

    ``helper_cache`` is keyed by module path: one api client is imported by
    every page, so it is scanned once rather than once per importer.

    A call that names no literal resource but whose path is read off an entry
    of an imported metadata object adds that property's values to ``traced``;
    one whose path is read off a component prop adds ``(prop, property)`` to
    ``prop_reads``, for the files that render the component to resolve. Text
    after such a read (`?owner=`, `bulk/`) keeps the path the entry's. One whose
    path is a whole prop that keys nothing (`api.post(endpoint, body)`) adds it to ``handed``.
    """
    calls = []
    bindings = _import_targets(rel, source, files)
    # The import statements name every helper without using it; a comment or a
    # string (`const mode = 'create'`) uses none. Arguments are read from ``uses``.
    uses = _IMPORT_RE.sub(lambda m: _KEEP_NEWLINES_RE.sub(" ", m.group()), source)
    code = _blank_literals(uses)
    for binding, spec in _import_bindings(source):
        target = next((c for c in _resolve_import(rel, spec) if c in files), None)
        if target is None or not _POST_RE.search(files[target]):
            continue
        if _GENERATED_CLIENT_RE.search(target):
            typed = _typed_client_creates(binding, code, files[target])
            if typed:
                calls.extend(typed)
                used_clients.add(target)
            continue
        if target not in helper_cache:
            helper_cache[target] = {**_post_helper_paths(files[target]), **_hook_post_paths(files[target])}
        helpers = helper_cache[target]
        if not helpers:
            continue
        construct = r"new\s+" + re.escape(binding) + r"\s*\("
        # An instance of an imported class: the name it is assigned to, or the `new` expression itself.
        receivers = [re.escape(binding), construct + r"[^()]*\)",
                     *(re.escape(name) for name in re.findall(r"([A-Za-z_$][\w$]*)\s*=\s*" + construct, code))]
        # ``chain``: members between the binding and the helper (``api.tag.create(``), the
        # resource of a client built by a factory (``tag: crud('tag')``).
        pattern = re.compile(
            r"\b(?:" + "|".join(receivers) + r")(?P<chain>(?:\s*\.\s*[\w$]+)*?)\s*(?:\.\s*(?P<member>[\w$]+))?\s*"
            r"(?:\((?P<args>[^()]{0,200})|(?![\w$.(]))")
        for call in pattern.finditer(code):
            name = call.group("member") or binding
            # Uncalled, a capitalised name is a component or class used as a value, not a create;
            # and only a value handed over (`fn: x`, `={x}`, `(x)`) is a use, not a word in JSX text.
            handed_over = ((call.group("member") or name[:1].islower())
                           and _previous_char(code, call.start()) in _REFERENCE_BEFORE
                           and _REFERENCE_AFTER_RE.match(code, call.end()))
            if name in helpers and (call.group("args") is not None or handed_over):
                args = uses[call.start("args"):call.end("args")] if call.group("args") is not None else ""
                head = _leading_expression(args)
                read = _prop_read(head, code) if args else None
                entries = _metadata_values(head, code, bindings, files) if args else []
                if helpers[name] or read or entries:
                    calls.append(helpers[name] or None)
                else:
                    # A helper that posts to the path it is given: `api.post(`/checkout/${cartId}/`, body)`.
                    path = _argument_path(uses, code, call.start("args")) if args and helpers[name] is None else None
                    calls.append(_STRING_ARG_RE.findall(args) or re.findall(r"[\w$]+", call.group("chain"))
                                 or ([path] if path else None))
                used_clients.add(target)
                if calls[-1] is None and traced is not None:
                    traced += entries
                    if read and prop_reads is not None:
                        prop_reads.append(read)
                    # A prop that indexes nothing (`ENTITIES[entity]` does) is no key of an imported config.
                    whole = head.strip()
                    if (handed is not None and re.fullmatch(r"[A-Za-z_$][\w$]*", whole)
                            and re.search(_PROPS_PARAMETER.format(name=re.escape(whole)), code)
                            and not re.search(r"\[\s*" + re.escape(whole) + r"\s*\]", code)):
                        handed.append(whole)
    return calls


def _typed_client_creates(binding: str, code: str, client: str) -> list:
    """Creates issued through one object of the generated typed client: ``taskApi.create(...)``
    creates what that object's create route posts to, and no other function of it creates."""
    block = re.search(r"export\s+const\s+" + re.escape(binding) + r"\s*=\s*\{(?P<body>[\s\S]*?)\n\};", client)
    path = block and re.search(r"/\*\*\s*POST\s+(\S+)\s*\*/\s*create\s*:", block.group("body"))
    if path is None:
        return []
    calls = re.finditer(r"(?<![\w$.])(?:\(\s*)?" + re.escape(binding) + r"(?:\s+as\s+[\w$<>\[\]]+\s*\))?"
                        r"\s*\.\s*(?:create|bulkCreate)\s*\(", code)
    return [[path.group(1)] for _ in calls]


def _table_block_create_wired(content: str, cls: str) -> bool:
    """True if a <TableBlock> in ``content`` is a wired create form for ``cls``.

    dataBinding.entity must equal ``cls`` (case-insensitive) with a real
    endpoint, and options.formColumns must carry at least one entry. A
    file can embed TableBlocks for other entities (a lookup column's own
    ``entity`` key, or an unrelated table on the same page) — scoping to
    each tag's own dataBinding, rather than a whole-file search, is what
    keeps this attributed to the right entity.
    """
    for tag in _TABLE_BLOCK_RE.findall(content):
        db_match = _DATA_BINDING_RE.search(tag)
        if not db_match:
            continue
        entity_match = _ENTITY_RE.search(db_match.group(1))
        endpoint_match = _ENDPOINT_RE.search(db_match.group(1))
        if not entity_match or not endpoint_match:
            continue
        if entity_match.group(1).strip().lower() != cls.lower():
            continue
        if not endpoint_match.group(1).strip():
            continue
        if _NONEMPTY_FORM_COLUMNS_RE.search(tag):
            return True
    return False


def _entity_forms(name: str) -> list[str]:
    """Lowercase spellings a generated app plausibly uses for a class."""
    low = name.lower()
    forms = {low, f"{low}s"}
    if low.endswith("y"):
        forms.add(f"{low[:-1]}ies")
    if low.endswith("s"):
        forms.add(f"{low}es")
    return sorted(forms, key=len, reverse=True)


@functools.lru_cache(maxsize=1024)
def _forms_re(forms: tuple[str, ...], whole_word: bool) -> re.Pattern:
    # A form must end its word: "running" is not Run, "username" is not User.
    # A whole word also keeps `RunStatus` / `RUN_STATUS` from declaring Run.
    alternatives = "(?:" + "|".join(map(re.escape, forms)) + ")"
    if whole_word:
        return re.compile(r"(?<![A-Za-z0-9_])" + alternatives + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    return re.compile(alternatives + r"(?-i:(?![a-z0-9]))", re.IGNORECASE)


def _mentions(text: str, forms: list[str], whole_word: bool = False) -> bool:
    return _forms_re(tuple(forms), whole_word).search(text) is not None


def _config_declarations(source: str) -> str:
    """The part of an entity config module that declares entities.

    An entry's own key, path and label declare it. Comments, multi-word labels
    (`'Dataset Version'`) and every list nested inside an entry (`fields: [...]`,
    whose `ref`s name other entities) are blanked, and so is an entry that offers no
    create form (`allowCreate: false`, `readOnly: true`).
    """
    source = _blank_prose(_without_uncreatable_entries(source))
    blanked = _blank_literals(source)
    spans, depth, start, outer = [], 0, None, 0
    for i, char in enumerate(blanked):
        if char in "([{":
            if char == "[" and depth >= 2 and start is None:
                start, outer = i, depth
            depth += 1
        elif char in ")]}":
            depth -= 1
            if start is not None and depth == outer:
                spans.append((start, i + 1))
                start = None
    for s, e in reversed(spans):
        source = source[:s] + _KEEP_NEWLINES_RE.sub(" ", source[s:e]) + source[e:]
    return source


def _creates(args: list[str] | None, forms: list[str]) -> bool:
    """Whether a create call can be this entity's.

    An unnamed target (a variable) can be; a literal one must name the entity:
    `api.post('/stockcheck/', ...)` on a page that lists products in a dropdown
    is not Product's create. Separators are ignored, so `/order-lines/` names
    OrderLine.
    """
    return args is None or _mentions(re.sub(r"[-_\s]", "", " ".join(args)), forms)


def _gui_creatable_classes(gui_model) -> set[str] | None:
    """Classes a GUI model offers to create: a table with action buttons, a form
    or a create button bound to them. None when no screen binds a class (the
    editor's default empty GUI), which scopes nothing."""
    creatable: set[str] = set()
    binds_a_class = False
    pending = [element for module in gui_model.modules or () for screen in module.screens or ()
               for element in (getattr(screen, "view_elements", None) or ())]
    while pending:
        element = pending.pop()
        pending.extend(getattr(element, "view_elements", None) or ())
        binding = getattr(element, "data_binding", None)
        bound = getattr(binding, "domain_concept", None)
        entity = getattr(element, "entity_class", None)
        binds_a_class = binds_a_class or bound is not None or entity is not None
        if bound is not None and (getattr(element, "action_buttons", False) or isinstance(element, Form)):
            creatable.add(bound.name)
        if (isinstance(element, Button) and entity is not None
                and element.actionType in (ButtonActionType.Create, ButtonActionType.Add)):
            creatable.add(entity.name)
    return creatable if binds_a_class else None


def build_acceptance_matrix(
    output_dir: str,
    domain_model,
    endpoint_manifest: str = "",
    snapshot_dir: str = ".besser_snapshot",
    gui_model=None,
) -> dict[str, dict[str, bool | None]] | None:
    """Compute {class name: {route, page, create}} for the workspace.

    A ``None`` cell is not a finding: the signal was not measured (``route`` on
    an unparsed stack) or does not apply (``create`` of an abstract class, or,
    with a ``gui_model`` whose screens bind a class, of a class they offer no
    way to create).

    ``endpoint_manifest`` is the statically parsed route listing (from
    ``prompt_builder.build_endpoint_manifest``); pass it in when the
    caller already built one, else it is rebuilt here.

    Returns None when there is no domain model / no classes.
    """
    if domain_model is None:
        return None
    try:
        classes = [c.name for c in domain_model.get_classes() if getattr(c, "name", None)]
    except Exception:
        return None
    if not classes:
        return None
    abstract = set(abstract_entities(domain_model))
    creatable = _gui_creatable_classes(gui_model) if gui_model is not None else None
    not_created = set(classes) - creatable if creatable is not None else set()

    if not endpoint_manifest:
        try:
            from besser.spec_driven_agent.agent.prompt_builder import build_endpoint_manifest
            endpoint_manifest = build_endpoint_manifest(output_dir) or ""
        except Exception:
            endpoint_manifest = ""
    manifest_low = endpoint_manifest.lower()

    # Collect frontend files once: (rel path, content).
    frontend_files: list[tuple[str, str]] = []
    for root, dirs, files in walk_plain(output_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith(_FRONTEND_EXTS):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, output_dir).replace("\\", "/")
            if rel.startswith(snapshot_dir) or rel.startswith(".besser_"):
                continue
            try:
                if os.path.getsize(fpath) > 1_000_000:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    frontend_files.append((rel, f.read()))
            except Exception:
                continue

    by_path = dict(frontend_files)
    helper_cache: dict[str, dict] = {}
    used_clients: set[str] = set()
    traced: dict[str, list[str]] = {rel: [] for rel, _ in frontend_files}
    prop_reads: dict[str, list[tuple[str, str]]] = {rel: [] for rel, _ in frontend_files}
    handed: dict[str, list[str]] = {rel: [] for rel, _ in frontend_files}
    client_calls = {rel: _client_create_calls(rel, content, by_path, helper_cache, used_clients,
                                              traced[rel], prop_reads[rel], handed[rel])
                    for rel, content in frontend_files}
    imports = {rel: _import_targets(rel, content, by_path) for rel, content in frontend_files}
    # A component handing its own prop on to one that reads the create path off
    # it reads that path too; repeated until no chain grows.
    changed = True
    while changed:
        changed = False
        for rel, content in frontend_files:
            for binding, target in imports[rel].items():
                for prop, entry_property in list(prop_reads[target]):
                    for own in _forwarded_props(content, binding, prop):
                        if (own, entry_property) not in prop_reads[rel]:
                            prop_reads[rel].append((own, entry_property))
                            changed = True
    # A create path read off a prop is that of the entry each rendering file hands the component.
    for rel, content in frontend_files:
        bindings = imports[rel]
        for binding, target in bindings.items():
            for prop, entry_property in prop_reads[target]:
                traced[rel] += _rendered_entry_values(content, binding, prop, entry_property, bindings, by_path)
    # A shared client's own POSTs count where its helpers are called, not as
    # creates of whatever its other helpers (`getMetrics`) happen to name.
    # A module other files call into is such a client even when none of those
    # calls is a create; a page imported to be rendered or routed is not.
    code = {rel: _blank_literals(content) for rel, content in frontend_files}
    clients = used_clients | {
        target for rel, content in frontend_files for binding, spec in _import_bindings(content)
        for target in [next((c for c in _resolve_import(rel, spec) if c in by_path), None)]
        if target in helper_cache
        and not re.search(r"</|/>", code[target])  # a closing tag: renders JSX, a component
        and re.search(r"(?<![\w$.])" + re.escape(binding) + r"\s*[.(]", code[rel])
        and not re.search(r"<\s*" + re.escape(binding) + r"\b", code[rel])
    } | {rel for rel in by_path if _GENERATED_CLIENT_RE.search(rel)}
    create_calls = {rel: client_calls[rel] + ([] if rel in clients else _direct_create_calls(content))
                    for rel, content in frontend_files}
    # Generic pages (``EntityList entity={key}``): the create names its resource
    # through a variable and the component's code names no class of the model
    # (its comments may: ``e.g. "/brand/"``). A file that mentions an entity and
    # renders one as JSX creates that entity through it; importing the api
    # client is not rendering a page. ``LoanList`` with a variable-argument
    # create is not generic: crediting it to every entity App.jsx lists would
    # hide their missing creates. Testing which entity it was given
    # (`config.key === 'booking'`) is what a shared form does, so it names none,
    # nor does the block it runs for that entity; neither does reading
    # `err.message` when the model has a Message, nor holding a `tagId`.
    all_forms = [f for name in classes for f in _entity_forms(name)]
    generic_creators = {
        rel for rel, calls in create_calls.items()
        if any(args is None for args in calls)
        and not _mentions(rel.rsplit("/", 1)[-1], all_forms)
        and not _names_an_entity(_blank_prose(by_path[rel]), all_forms)
    }
    # A generic creator may read its entities from a config module it imports
    # (`entities.js`); that module, not the creator's own code, declares them.
    # An entry's nested `actions: [{ method: 'POST' }]` is data and is blanked
    # with the other nested lists; a POST left over makes the module a client.
    # A creator that posts to a path its parent hands over (`api.post(endpoint)`)
    # is not configured by what it imports: the parent decides what it creates.
    config_text = {
        rel: "" if handed[rel] else " ".join(text for text in (
            _config_declarations(by_path[target]) for target in (
                next((c for c in _resolve_import(rel, spec) if c in by_path), None)
                for _, spec in _import_bindings(by_path[rel]))
            if target and not re.search(r"\bfetch\s*\(", by_path[target]))
            if not _POST_RE.search(text))
        for rel in generic_creators
    }
    # A configured creator is credited through its config alone: the page that
    # renders it mentioning `user` (the signed-in one) declares nothing. Next
    # to such a config a message the creator shows (`'Add a product first.'`)
    # names no entity; without one, the message ties the form to that entity.
    unconfigured = {rel for rel in generic_creators
                    if not _mentions(config_text[rel], all_forms, whole_word=True)
                    and not _names_an_entity(_blank_comments(by_path[rel]), all_forms)}
    renders_generic_creator = {
        rel: any(next((c for c in _resolve_import(rel, spec) if c in by_path), None) in unconfigured
                 and re.search(r"<\s*" + re.escape(binding) + r"\b", content)
                 for binding, spec in _import_bindings(content))
        for rel, content in frontend_files
    }

    matrix: dict[str, dict[str, bool | None]] = {}
    for cls in sorted(classes):
        forms = _entity_forms(cls)
        # None, not False, when the manifest is empty: build_endpoint_manifest
        # parses FastAPI decorators out of .py files only, so an Express,
        # Django, Spring or axum backend yields "" and every entity would read
        # "no backend REST route" for a route the app genuinely serves. An
        # unparsed stack is unmeasured, not empty.
        route = any(f in manifest_low for f in forms) if manifest_low else None
        page = False
        create = False
        for rel, content in frontend_files:
            # A create whose path is read off this entity's metadata entry.
            traced_create = any(_creates([value], forms) for value in traced[rel])
            # An import line names a module, not the entity (`../api/generated/client` is no Client).
            about = (_mentions(rel.rsplit("/", 1)[-1], forms) or _mentions(_IMPORT_RE.sub("", content), forms)
                     or _mentions(config_text.get(rel, ""), forms, whole_word=True) or traced_create)
            if not about:
                continue
            page = True
            # A create that names another resource literally is that resource's
            # create, not this entity's: `api.create('order', ...)` on a page
            # that merely lists clerks in a dropdown proves nothing about Clerk.
            if (traced_create or any(_creates(args, forms) for args in create_calls[rel])
                    or renders_generic_creator[rel]
                    or _table_block_create_wired(content, cls)):
                create = True
                break
        # An abstract class is created through its subclasses, and the GUI model
        # scopes which classes the screens create: no create form is expected.
        matrix[cls] = {"route": route, "page": page,
                       "create": None if cls in abstract or cls in not_created else create}
    return matrix


def matrix_issues(matrix: dict[str, dict[str, bool | None]] | None) -> list[str]:
    """Render missing matrix cells as advisory issue strings."""
    if not matrix:
        return []
    issues: list[str] = []
    for cls, cells in matrix.items():
        # `is False` and not falsiness: None means the signal was never
        # collected or does not apply, and neither is a finding.
        missing = [k for k in ("route", "page", "create") if cells.get(k) is False]
        if not missing:
            continue
        detail = {
            "route": "no backend REST route",
            "page": "no frontend page/component references it",
            "create": "no frontend create path found (no POST call, no api-client create resolved to it, and no TableBlock bound to it with editable form fields)",
        }
        issues.append(
            "acceptance: entity "
            + cls
            + " — "
            + "; ".join(detail[m] for m in missing)
        )
    return issues
