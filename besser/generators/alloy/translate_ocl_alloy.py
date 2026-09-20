"""
translate_ocl_alloy.py

Translates OCL constraints (on a BESSER/BUML model) to Alloy facts.

Main entry point:
    ocl_to_alloy(inherits_from, data, ocl, context_name, estado, enums) -> str

Internal flow:
    OCL (str)
      └─ parse_ocl_expression()   →  ANTLR tree
      └─ tokenize_tree()          →  list of tokens (type, value)
      └─ write_prefix_ocl()       →  tokens with resolved class prefixes
      └─ parse_predicate()        →  own AST
      └─ ast_to_alloy()           →  string (Alloy fact)
"""

# ── Standard library ─────────────────────────────────────────────────────────
import re
from dataclasses import dataclass, field

# ── Third-party ──────────────────────────────────────────────────────────────
from antlr4 import CommonTokenStream, InputStream
from antlr4.tree.Tree import TerminalNode as TN
from dateutil import parser as dateutil_parser

# ── BESSER / BUML ─────────────────────────────────────────────────────────────
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.generators.alloy.date_ops import (
    DATES_DICT,
    DateOpsRegistry,
    encode_date,
)
from besser.generators.alloy.string_ops import StringOpError, StringOpsRegistry

# ── Types ─────────────────────────────────────────────────────────────────────
Token = tuple[str, str]

# ══════════════════════════════════════════════════════════════════════════════
# 1. TRANSLATION STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TranslatorState:
    """
    Accumulates state during the translation of OCL constraints for one model.

    A single instance is shared across every constraint of a model (see
    ``AlloyGenerator.generate``), so anything that is model-wide rather than
    constraint-wide — such as the enumeration catalog — belongs here instead
    of being re-passed on every call.

    Attributes:
        cont_select:     Counter for generating unique select/reject function names.
        dates:          ``dMMDDYYYY`` sig ids already emitted, so identical
                         date literals are only declared once across the model.
        strings:        Unique string literals found so far, in first-seen
                         order.  Mirrors ``dates``: every literal is declared
                         only once across the whole model (see
                         :meth:`register_string` and
                         :func:`build_string_sigs`).  ``''`` (the empty
                         string) is a valid entry and gets its own sig.
        buffer_pred_aux: Auxiliary predicates (select, reject) accumulated so far.
        is_set_origin:   True when the current collection is a flat set (not a relation).
        enums:           ``{EnumName: {Literal, ...}}`` catalog for the current model,
                         set once via :meth:`set_enums`.
        data:            ``{ClassName: ['field:Type', ...]}`` attribute map of the
                         model, used to detect date-typed attribute operands.
        string_ops:      OCL String operation registry used to translate
                         String method calls and to emit ``str_ops.als``.
        date_ops:        OCL Date comparison operation registry used to
                         translate ordered date comparisons and to emit
                         ``date.als``.
        maxseq:          Maximum length of the string sequences found when
                         processing string literals (see
                         :func:`process_string_types`).  Defaults to the
                         generator scope (5) and is only raised when a
                         longer string literal appears in a constraint.
    """

    cont_select: int = 0
    maxseq: int = 0
    dates: list = field(default_factory=list)
    strings: list = field(default_factory=list)
    buffer_pred_aux: list = field(default_factory=list)
    is_set_origin: bool = True
    enums: dict[str, set[str]] = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    string_ops: StringOpsRegistry = field(default_factory=StringOpsRegistry)
    date_ops: DateOpsRegistry = field(default_factory=DateOpsRegistry)
    _enum_token_index: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _string_name_index: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def init_constraint(self) -> None:
        """Reset mutable state before processing a new constraint.

        Deliberately does NOT touch ``enums`` and ``strings`` (+ their
        indexes): those are model-wide and set/accumulated once, not
        per-constraint.
        """
        self.is_set_origin = True
        self.buffer_pred_aux.clear()

    def write_aux_pred(self, text: str) -> None:
        """
        Appends *text* to the auxiliary-predicate buffer.
        """
        self.buffer_pred_aux.append(text)

    def read_aux_pred(self) -> str:
        """
        Returns all accumulated auxiliary predicates joined by newlines.
        """
        return "\n".join(self.buffer_pred_aux)

    def set_enums(self, enums: dict[str, set[str]] | None) -> None:
        """
        Registers the model's enumeration catalog and (re)builds its index.

        Building the ``ENUM_<Enum>_<Literal> -> <Enum>`` lookup is O(total
        literals) and only needs to happen once per model: every constraint
        in the same model shares the same ``enums`` mapping, so an identity
        check makes this a no-op on every call after the first instead of
        rebuilding — or rescanning — the catalog once per constraint.
        """
        if enums is None:
            enums = {}
        if enums is self.enums and self._enum_token_index:
            return
        self.enums = enums
        self._enum_token_index = {
            f"ENUM_{enum_name}_{literal}": enum_name
            for enum_name, literals in enums.items()
            for literal in literals
        }

    def resolve_enum_token(self, token_value: str) -> str | None:
        """
        Returns the owning enum name for a normalized ``ENUM_...`` token, or ``None``.
        """
        return self._enum_token_index.get(token_value)

    def register_string(self, literal: str) -> str:
        """Registers a string *literal* on the model-wide catalog and returns its
        Alloy sig name.

        Identical literals — including the empty string ``''`` — are only
        registered once across the whole model (first-seen order), so they
        map to a single ``one sig StrN`` declaration shared by every
        constraint (see :func:`build_string_sigs`).  The generated ``StrN``
        name is always a valid Alloy identifier, unlike the raw literal
        (e.g. ``''``, ``'good morning'``).

        Also raises ``state.maxseq`` to the literal length when it is the
        longest seen so far.
        """
        name = self._string_name_index.get(literal)
        if name is None:
            name = f"Str{len(self.strings)}"
            self._string_name_index[literal] = name
            self.strings.append(literal)
            self.maxseq = max(self.maxseq, len(literal))
        return name


class EnumReferenceError(ValueError):
    """
    Raised when an OCL constraint references an unknown enumeration type
    or a literal that is not declared in the target enumeration.

    This prevents the Alloy generator from emitting facts that reference
    undefined Alloy signatures (e.g. after a literal was renamed, removed
    or its casing was changed), which Alloy would otherwise reject with a
    cryptic "name cannot be found" parse error.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 2. OWN AST NODES
# ══════════════════════════════════════════════════════════════════════════════

class Literal:
    """Numeric or string literal."""

    def __init__(self, val: str):
        self.val = val


class BLiteral:
    """OCL boolean literal (true / false)."""

    def __init__(self, val: str):
        self.val = "True" if val.lower() == "true" else "False"


class Var:
    """Simple variable (self, identifier without navigation)."""

    def __init__(self, name: str):
        self.name = name


class Enumeration:
    """Enumeration value."""

    def __init__(self, name: str):
        self.name = name


class Nav:
    """OCL navigation of the form a.b.c (two or more parts)."""

    def __init__(self, parts: list[str]):
        self.parts = parts


class Call:
    """OCL operation call: expr.op(args) or expr->op(args)."""

    def __init__(self, expr, callname: str, args=None):
        self.expr = expr
        self.callname = callname
        self.args = args or []


class IfThenElse:
    """OCL conditional expression: if cond then t else e endif."""

    def __init__(self, cond, then_expr, else_expr):
        self.cond = cond
        self.then_expr = then_expr
        self.else_expr = else_expr


class BinaryOp:
    """Binary operator (=, !=, <, >, and, or, implies, +, -, ...)."""

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class UnaryOp:
    """Unary operator: ``not x`` or ``-x``."""

    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand


class IteratorOp:
    """OCL iterator: col->forAll/exists/select/reject/collect(v | expr)."""

    def __init__(self, kind: str, varnames: list[str], collection, expr):
        self.kind = kind
        self.varnames = varnames
        self.collection = collection
        self.expr = expr
        self.generated: bool = False  # internal flag for select/reject


# ══════════════════════════════════════════════════════════════════════════════
# 3. OCL PARSER (ANTLR) → TOKEN LIST
# ══════════════════════════════════════════════════════════════════════════════

def parse_ocl_expression(ocl_input: str):
    """
    Invokes ANTLR and return ``(tree, parser)``.
    """
    input_stream = InputStream(ocl_input)
    lexer = BOCLLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    ocl_parser = BOCLParser(token_stream)
    tree = ocl_parser.expression()
    return tree, ocl_parser


def tokenize_tree(tree, parser) -> list[Token]:
    """
    Walks the ANTLR tree and produces a list of ``Token`` (type, value).

    At the end normalises the pattern ``Class::EnumVal`` into a single
    ``('enum', 'ENUM_Class_Val')`` token.
    """
    tokens: list[Token] = []

    _ITERATORS = {"forall", "exists", "reject", "select", "collect"}
    _CALLS = {
        "including", "excluding", "oclistypeof", "ocliskindof",
        "oclisundefined", "asset", "isempty", "notempty",
        "includesall", "includes", "excludes", "excludesall", "closure",
        "union", "intersection",
    }
    _OPERATORS = {">", "<", ">=", "<=", "=", "+", "-", "*", "/", "and", "or", "not", "implies"}
    _PUNCTUATION = {"(", ")", "{", "}", ",", ":"}

    def _classify_terminal(txt: str) -> None:
        """
        Classifies a single ANTLR terminal and appends the corresponding token.
        """
        if txt == "::":
            tokens.append(("::", "::"))
            return

        if tokens and tokens[-1][0] == "::":
            if txt.lower() == "allinstances":
                tokens.append(("allInstances", "allInstances"))
            else:
                tokens.append(("enum", txt))
            return

        if txt == ".":
            tokens.append(("dot", "."))
            return

        if txt in ("->", "→"):
            tokens.append(("arrow", "->"))
            return

        if txt in _PUNCTUATION:
            tokens.append((txt, txt))
            return

        if txt in _OPERATORS:
            tokens.append(("operator", txt))
            return

        if txt == "<>":
            tokens.append(("operator", "!="))
            return

        if txt == "|":
            tokens.append(("pipe", "|"))
            return

        if txt.isdigit():
            tokens.append(("literal", txt))
            return

        low = txt.lower()

        if low in ("true", "false"):
            tokens.append(("bliteral", low))
            return

        if (txt.startswith("'") and txt.endswith("'")) or (txt.startswith('"') and txt.endswith('"')):
            tokens.append(("literal", txt.lower()))
            return

        if low == "self":
            tokens.append(("self", "self"))
            return

        if low == "null":
            tokens.append(("null", "null"))
            return

        if low in ("if", "then", "else", "endif"):
            tokens.append((low, low))
            return

        if low in _ITERATORS:
            tokens.append(("iterator_op", low))
            return

        if low in _CALLS:
            tokens.append(("call", low))
            return

        if low == "allinstances":
            tokens.append(("allInstances", "allInstances"))
            return

        tokens.append(("id", txt))

    def walk(node) -> None:
        if isinstance(node, TN):
            txt = node.getText()
            if txt and txt.strip():
                _classify_terminal(txt)
            return
        for i in range(node.getChildCount()):
            walk(node.getChild(i))

    walk(tree)
    return _normalize_enum_pattern(_normalize_dot_calls(tokens))


def _normalize_dot_calls(tokens: list[Token]) -> list[Token]:
    """
    Promotes an ``id`` directly followed by ``(`` to a ``call``.  The OCL grammar
    only produces ``ID LPAREN`` after a dot (``dotMethodCall``/``dotSize``), so
    this is always a method call (e.g. ``self.title.size()``), never attribute
    navigation.
    """
    return [
        ("call", tokens[i][1])
        if tokens[i][0] == "id" and i + 1 < len(tokens) and tokens[i + 1][0] == "("
        else tokens[i]
        for i in range(len(tokens))
    ]


def _normalize_enum_pattern(tokens: list[Token]) -> list[Token]:
    """
    Collapses ``('id','Class') ('::', '::') ('enum','Val')`` into ``('enum', 'ENUM_Class_Val')``.
    """
    result: list[Token] = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i][0] == "id"
            and tokens[i + 1][0] == "::"
            and tokens[i + 2][0] == "enum"
        ):
            result.append(("enum", f"ENUM_{tokens[i][1]}_{tokens[i + 2][1]}"))
            i += 3
        else:
            result.append(tokens[i])
            i += 1
    return result


def _describe_unknown_enum_token(
    token_value: str,
    enums: dict[str, set[str]],
    context_name: str,
) -> EnumReferenceError:
    """
    Builds a precise :class:`EnumReferenceError` for an unrecognized enum token.

    Only called on the error path (the token already failed the O(1) index
    lookup), so the linear scan over ``enums`` here costs nothing in the
    common case — it exists purely to tell "unknown enum type" apart from
    "known enum, unknown literal" for the error message.
    """
    for enum_name, literals in enums.items():
        prefix = f"ENUM_{enum_name}_"
        if token_value.startswith(prefix):
            literal = token_value[len(prefix):]
            return EnumReferenceError(
                f"OCL constraint (context {context_name}) references enumeration "
                f"literal '{enum_name}::{literal}', but '{literal}' is not a value "
                f"of enumeration '{enum_name}'. Available values: "
                f"{', '.join(sorted(literals)) or '(none)'}."
            )

    display = token_value[len("ENUM_"):].replace("_", "::", 1) if "_" in token_value else token_value
    return EnumReferenceError(
        f"OCL constraint (context {context_name}) references "
        f"'{display}', but the referenced type is not a known enumeration."
    )


def validate_enum_references(
    toks: list[Token],
    state: "TranslatorState",
    context_name: str,
) -> None:
    """
    Validates OCL enumeration references against the model's enumerations.

    OCL references of the form ``EnumType::Literal`` are normalized to
    ``ENUM_<EnumType>_<Literal>`` tokens.  This function checks that the
    referenced enumeration type exists and that the literal is one of its
    declared values, raising :class:`EnumReferenceError` otherwise.  This
    stops the Alloy generator from emitting facts that reference undefined
    signatures (e.g. after a literal was renamed or removed), which Alloy
    would otherwise reject with a cryptic "name cannot be found" error.

    The enum catalog lives on ``state`` (see ``TranslatorState.set_enums``),
    so each token is checked with a single dict lookup instead of scanning
    every enumeration for a prefix match.

    Args:
        toks:         Token list produced by :func:`tokenize_tree`.
        state:       Shared translator state carrying the enum catalog.
                      An empty/unset catalog disables the check (backward
                      compatibility with callers that don't pass ``enums``).
        context_name: Name of the OCL constraint context class, used only
                      for error messages.

    Raises:
        EnumReferenceError: If an enum type or literal is unknown.
    """
    if not state.enums:
        return

    for t, v in toks:
        if t != "enum":
            continue
        if state.resolve_enum_token(v) is None:
            raise _describe_unknown_enum_token(v, state.enums, context_name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASS PREFIX RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _search_field(subject_class: str, field: str, data: dict) -> str:
    """Returns the type of *field* in *subject_class*, or ``''`` if not found."""
    for entry in data.get(subject_class, []):
        field_name, field_type = entry.split(":", 1)
        if field_name == field:
            return field_type
    return ""


def _iter_parent_chain(subject_class: str, inherits_from: dict):
    """Yields *subject_class* and then its first-parent chain until the root marker."""
    curr_class = subject_class
    while curr_class:
        yield curr_class
        parents = inherits_from.get(curr_class, "_")
        if parents in ("_", None, []):
            break
        curr_class = parents[0]


def _resolve_field_with_inheritance(subject_class: str, subject_field: str, data: dict, inherits_from: dict):
    """Returns ``(owner_class, field_type)`` for *subject_field* searching *subject_class* and parents."""
    for candidate in _iter_parent_chain(subject_class, inherits_from):
        field_type = _search_field(candidate, subject_field, data)
        if field_type:
            return candidate, field_type
    return "", ""


def _record_vars_iterator(
    toks: list[Token],
    i_iter: int,
    curr_type: str,
    var_types: dict[str, str],
) -> dict[str, str]:
    """
    Records iterator variables (forAll/exists/select/reject/collect).

    Supports both untyped and typed declarations, e.g.:
        forAll(e | ...)
        forAll(e: Edge | ...)
        exists(a, b: Node | ...)
    """
    if i_iter + 1 >= len(toks) or toks[i_iter + 1][0] != "(":
        return {}

    scope: dict[str, str] = {}
    pending_vars: list[str] = []

    j = i_iter + 2
    while j < len(toks):
        t, v = toks[j]
        if t in {"pipe", ")"}:
            break

        if t == "id":
            prev_type = toks[j - 1][0] if j - 1 >= 0 else None
            if prev_type == ":":
                # Type annotation: e.g. e: Edge | ...
                for var in pending_vars:
                    scope[var] = v
                pending_vars.clear()
            else:
                # Iterator variable candidate: e.g. e, x, y
                pending_vars.append(v)

        j += 1

    for var in pending_vars:
        scope[var] = curr_type

    var_types.update(scope)
    return scope


def write_prefix_ocl(
    toks: list[Token],
    data: dict,
    inherits_from: dict,
    context_name: str,
) -> list[Token]:
    """
    Resolves class prefixes for each attribute/relation identifier.

    For example, if ``name`` is an attribute of ``Person``, the token
    ``'name'`` becomes ``'Person_name'``.  Navigates the inheritance
    hierarchy when the field is not found in the current class.

    Args:
        toks:         Token list produced by :func:`tokenize_tree`.
        data:         ``{ClassName: ['field:Type', ...]}`` attribute map.
        inherits_from: ``{ClassName: [Parent, ...] | '_'}`` hierarchy map.
        context_name: Name of the OCL context class.

    Returns:
        The mutated token list with class-prefixed identifiers.
    """
    subject_type = context_name
    is_traversed = False
    paren_depth = 0
    var_types: dict[str, str] = {}
    scope_stack: list[dict[str, str]] = [{}]
    iterator_scope_depths: list[int] = []

    def _lookup_var_type(name: str) -> str | None:
        for scope in reversed(scope_stack):
            if name in scope:
                return scope[name]
        return None

    def _collapse_possible_allInstances(toks: list[Token]) -> list[Token]:
        """
        Collapses ``Class.allInstances()`` or ``Class::allInstances()``
        into a single token.  This is a special case of navigation that is
        not a field, so it does not get prefixed with the class name.
        Instead, it becomes a single token of type ``class`` with value
        ``ClassName`` (which denotes the Alloy signature / set of instances).
        """
        result: list[Token] = []
        i = 0
        while i < len(toks):
            if (
                i + 4 < len(toks)
                and toks[i][0] == "id"
                and toks[i + 1][0] in ("dot", "::")
                and toks[i + 2][0] == "allInstances"
                and toks[i + 3][0] == "("
                and toks[i + 4][0] == ")"
            ):
                result.append(("class", toks[i][1]))
                i += 5
            else:
                result.append(toks[i])
                i += 1
        return result

    toks = _collapse_possible_allInstances(toks)
    for i, (t, v) in enumerate(toks):
        if t == "(":
            paren_depth += 1
            continue

        if t == ")":
            paren_depth -= 1
            while iterator_scope_depths and paren_depth < iterator_scope_depths[-1]:
                iterator_scope_depths.pop()
                scope_stack.pop()
            continue

        if t == "dot":
            is_traversed = True
            continue

        if t == "null" and v.lower() == "null":
            toks[i] = (t, "null")
            continue

        if t == "self":
            subject_type = context_name
            is_traversed = False
            continue

        if t == "class":
            subject_type = v
            is_traversed = False
            continue

        if t == "iterator_op":
            iter_scope = _record_vars_iterator(toks, i, subject_type, var_types)
            scope_stack.append(iter_scope)
            iterator_scope_depths.append(paren_depth + 1)
            is_traversed = False
            continue

        if t == "call" and v == "closure":
            is_traversed = True
            continue

        if t in {"call", "if", "then", "else", "endif"}:
            is_traversed = False
            continue

        if t == "id" and not is_traversed:
            siguiente = toks[i + 1][0] if i + 1 < len(toks) else None
            tipo_var = _lookup_var_type(v)
            if siguiente == "dot" and tipo_var:
                subject_type = tipo_var
            continue

        if t == "id" and is_traversed:
            is_traversed = False
            owner, tipo_campo = _resolve_field_with_inheritance(subject_type, v, data, inherits_from)
            if not owner:
                continue

            toks[i] = (t, f"{owner}_{v}")
            # If the field type is a known class, continue navigation from that class.
            # Otherwise remain in the class where the field is declared.
            subject_type = tipo_campo if tipo_campo in data else owner

    return toks


# ══════════════════════════════════════════════════════════════════════════════
# 5. TOKEN LIST → OWN AST
# ══════════════════════════════════════════════════════════════════════════════

def parse_predicate(tokens: list[Token]):
    """
    Recursive-descent parser over the token list from :func:`tokenize_tree`.

    Returns the root node of the own AST.
    """
    pos = 0

    def peek(k: int = 0):
        idx = pos + k
        return tokens[idx] if idx < len(tokens) else None

    def consume(expected_type: str | None = None, expected_val: str | None = None):
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of tokens")
        tok = tokens[pos]
        pos += 1
        if expected_type and tok[0] != expected_type:
            raise ValueError(f"Expected type {expected_type!r}, found {tok}")
        if expected_val and tok[1] != expected_val:
            raise ValueError(f"Expected value {expected_val!r}, found {tok}")
        return tok

    def _collect_until(stop_type: str) -> list[Token]:
        """Consumes tokens until one of *stop_type* is found (exclusive)."""
        buf: list[Token] = []
        while peek() and peek()[0] != stop_type:
            buf.append(consume())
        return buf

    def _collect_args() -> list:
        """Consumes arguments of a call between already-opened parentheses."""
        depth = 1
        current: list[Token] = []
        args = []
        while True:
            t = consume()
            if t[1] == "(":
                depth += 1
            elif t[1] == ")":
                depth -= 1
                if depth == 0:
                    if current:
                        args.append(parse_predicate(current))
                    break
            elif t[1] == "," and depth == 1:
                args.append(parse_predicate(current))
                current = []
                continue
            current.append(t)
        return args

    def parse_primary():
        nonlocal pos
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end of tokens (primary)")

        # if-then-else
        if tok[0] == "if":
            consume("if")
            cond_toks = _collect_until("then")
            consume("then")
            then_toks = _collect_until("else")
            consume("else")
            else_toks = _collect_until("endif")
            consume("endif")
            return IfThenElse(
                parse_predicate(cond_toks),
                parse_predicate(then_toks),
                parse_predicate(else_toks),
            )

        if tok[0] == "bliteral":
            consume()
            return BLiteral(tok[1])

        if tok[0] == "literal":
            consume()
            return Literal(tok[1])

        if tok[0] in ("id", "self", "enum", "null", "class"):
            es_enum = tok[0] == "enum"
            parts = [tok[1]]
            consume()

            while peek() and peek()[0] == "dot":
                consume("dot")
                nxt = peek()
                if nxt and nxt[0] == "id":
                    consume("id")
                    parts.append(nxt[1])
                elif nxt and nxt[0] == "call":
                    break
                else:
                    raise ValueError(
                        "Alloy generation failed while translating an OCL constraint: "
                        "'allInstances()' is a type-level operation and can only be called "
                        "on a class/type name (e.g. 'Employee.allInstances()'), not on an "
                        "instance such as 'self'. Please review the OCL constraints in your "
                        "class diagram."
                    ) from ValueError(f"Unexpected token after '.': {nxt}")

            if es_enum and len(parts) > 1:
                raise ValueError(
                    f"Cannot navigate through enumeration literal: {'.'.join(parts)}"
                )

            if es_enum:
                node = Enumeration(parts[0])
            else:
                node = Nav(parts) if len(parts) > 1 else Var(parts[0])

            if peek() and peek()[0] == "dot":
                consume("dot")
            if peek() and peek()[0] == "call":
                callname = consume("call")[1]
                args = []
                if peek() and peek()[1] == "(":
                    consume("(")
                    args = _collect_args()
                node = Call(node, callname, args)

            while True:
                if peek() and peek()[0] == "arrow":
                    consume("arrow")
                    nxt = peek()
                    if nxt is None:
                        break

                    if nxt[0] == "iterator_op":
                        kind = consume("iterator_op")[1]
                        if not (peek() and peek()[1] == "("):
                            raise ValueError("Missing '(' after iterator_op")
                        consume("(")

                        varnames: list[str] = []
                        if peek() and peek()[0] == "id":
                            varnames.append(consume("id")[1])
                            while peek() and peek()[0] == ",":
                                consume(",")
                                if peek() and peek()[0] == "id":
                                    varnames.append(consume("id")[1])
                                else:
                                    raise ValueError("Expected identifier after ',' in iterator")

                        if peek() and peek()[0] == ":":
                            consume(":")
                            if peek() and peek()[0] == "id":
                                consume("id")  # type annotation ignored

                        if not (peek() and peek()[0] == "pipe"):
                            raise ValueError("Missing '|' in iterator_op")
                        consume("pipe")

                        depth = 1
                        inner_toks: list[Token] = []
                        while True:
                            t = consume()
                            if t[1] == "(":
                                depth += 1
                            elif t[1] == ")":
                                depth -= 1
                                if depth == 0:
                                    break
                            inner_toks.append(t)

                        node = IteratorOp(kind, varnames, node, parse_predicate(inner_toks))
                        continue

                    if nxt[0] in ("call", "id"):
                        callname = consume()[1]
                        args = []
                        if peek() and peek()[1] == "(":
                            consume("(")
                            args = _collect_args()
                        node = Call(node, callname, args)
                        continue

                    raise ValueError(f"Unexpected token after '->': {nxt}")

                if (
                    peek()
                    and peek()[0] == "dot"
                    and peek(1)
                    and peek(1)[0] == "call"
                ):
                    consume("dot")
                    callname = consume("call")[1]
                    args = []
                    if peek() and peek()[1] == "(":
                        consume("(")
                        args = _collect_args()
                    node = Call(node, callname, args)
                    continue

                break

            return node

        if tok[1] == "(":
            consume("(")
            sub: list[Token] = []
            depth = 1
            while True:
                t = consume()
                if t[1] == "(":
                    depth += 1
                elif t[1] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                sub.append(t)
            return parse_predicate(sub)

        raise ValueError(f"Unexpected token in parse_primary: {tok}")

    # ── Operator hierarchy (highest to lowest precedence) ─────────────────────

    def parse_unary():
        if peek() and peek()[0] == "operator" and peek()[1] in ("not", "-"):
            op = consume("operator")[1]
            return UnaryOp(op, parse_unary())
        return parse_primary()

    def parse_additive():
        left = parse_unary()
        while peek() and peek()[0] == "operator" and peek()[1] in ("+", "-"):
            op = consume("operator")[1]
            left = BinaryOp(op, left, parse_unary())
        return left

    def parse_compare():
        left = parse_additive()
        while peek() and peek()[0] == "operator" and peek()[1] in ("=", "!=", "<", ">", "<=", ">="):
            op = consume("operator")[1]
            left = BinaryOp(op, left, parse_additive())
        return left

    def parse_and():
        left = parse_compare()
        while peek() and peek()[0] == "operator" and peek()[1] == "and":
            op = consume("operator")[1]
            left = BinaryOp(op, left, parse_compare())
        return left

    def parse_or():
        left = parse_and()
        while peek() and peek()[0] == "operator" and peek()[1] in ("or", "implies"):
            op = consume("operator")[1]
            left = BinaryOp(op, left, parse_and())
        return left

    return parse_or()


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DATE_TYPES = {"date", "datetime", "time", "timedelta"}

_DATE_PATTERN = re.compile(
    r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$"
    r"|^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s+\d{4})?$",
    re.IGNORECASE,
)


def is_date(s: str) -> str | None:
    """
    Detects whether *s* is a date literal and returns its Alloy id (``dMMDDYYYY``) or ``None``.

    The whole content (after stripping single/double quotes) must be a date, so
    arbitrary strings that merely contain a date-like substring (e.g.
    ``'date 2024-01-01 x'``) are left untouched.
    """
    contents = s.strip().strip("'").strip('"').strip()
    if not _DATE_PATTERN.match(contents):
        return None
    try:
        curr_date = dateutil_parser.parse(contents)
        return encode_date(curr_date)
    except (ValueError, OverflowError):
        return None


def parse_date(s: str, state: TranslatorState) -> str:
    """
    Parses *s* as a date and records its ``dMMDDYYYY`` id on *state*.

    The ``one sig`` declarations and the ordering fact are emitted later by
    :func:`generate_dates_and_order`, which assigns the sequential ``DateN``
    sig names (``Date0``, ``Date1``, ...) to every date of the model — OCL
    literals are treated exactly like randomly generated dates. Duplicated
    literals are only recorded once across the whole model.

    Returns an empty string.

    Raises:
        ValueError: If *s* cannot be interpreted as a date.
    """
    contents = s.strip().strip("'").strip('"').strip()
    try:
        curr_date = dateutil_parser.parse(contents)
    except (ValueError, OverflowError):
        raise ValueError(f"Invalid date in OCL constraint: {s!r}") from None
    sig_id = encode_date(curr_date)
    if sig_id not in state.dates:
        state.dates.append(sig_id)
    return ""


_DATE_LITERAL_PATTERN = re.compile(r"\bd\d{8}\b")


def resolve_ocl_date_literals(constraints) -> None:
    """Rewrites translated OCL facts in place.

    Replaces every ``dMMDDYYYY`` literal id produced by :func:`is_date` /
    :func:`parse_date` with the sequential ``DateN`` sig name assigned by
    :func:`generate_dates_and_order` (via :data:`DATES_DICT`), so OCL date
    constants use exactly the same atoms as randomly generated dates.\n
    Unknown ids (not present in :data:`DATES_DICT`) are left untouched.
    """
    if not DATES_DICT:
        return
    name_by_id = {v: k for k, v in DATES_DICT.items()}
    for constraint in constraints:
        constraint.expression = _DATE_LITERAL_PATTERN.sub(
            lambda m: name_by_id.get(m.group(0), m.group(0)),
            constraint.expression,
        )


def _is_date_field(subject_field: str, data: dict) -> bool:
    """Returns ``True`` if *subject_field* references a date-typed attribute.

    Attribute references appear class-prefixed (``Class_field``) in the
    translated Alloy string (see :func:`write_prefix_ocl`), so they can be
    matched against the ``{ClassName: ['field:Type', ...]}`` *data* map.
    """
    if not data:
        return False
    for curr_class, fields in data.items():
        for curr_field in fields:
            field_name, field_type = curr_field.split(":", 1)
            if field_type in _DATE_TYPES and f"{curr_class}_{field_name}" in subject_field:
                return True
    return False


def _field_type_of(subject_field: str, data: dict) -> str:
    """Return the Alloy type of *subject_field* (e.g. ``self.Book_title`` → ``str``), or ``''``.

    The match is *exact* (full prefixed name at the end of the expression) so
    compound collection expressions that merely *contain* a field reference are
    not misclassified as that field's type.
    """
    for curr_class, fields in data.items():
        for curr_field in fields:
            field_name, field_type = curr_field.split(":", 1)
            pref = f"{curr_class}_{field_name}"
            if subject_field == pref or subject_field.endswith("." + pref):
                return field_type
    return ""


def _is_string_operand(subject: str, state: TranslatorState) -> bool:
    """Returns ``True`` when *subject* is a String operand: a ``str``-typed
    attribute or a quoted string literal that is not a date literal.

    Date literals are deliberately excluded because the date block in
    :func:`_translate_binaryop` runs before the string block and owns them.
    """
    if _field_type_of(subject, state.data) == "str":
        return True
    stripped = subject.strip()
    if (stripped.startswith("'") and stripped.endswith("'")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return is_date(stripped) is None
    return False


def process_string_types(input_string: str, state: TranslatorState | None = None) -> str:
    """Replaces quoted string literals in the assembled Alloy fact with their
    model-wide sig names and registers them on *state*.

    Every ``'literal'`` occurrence is substituted by the ``StrN`` sig name
    assigned via :meth:`TranslatorState.register_string`, so identical
    literals — including the empty string ``''`` — always reference the same
    single ``one sig``.  The sig declarations themselves are emitted once per
    model by :func:`build_string_sigs` (in ``strings.als``), not inline here.

    Also records on *state* the maximum length of the extracted sequences
    (``state.maxseq``), used later to bound the Alloy ``seq`` scope.
    """
    if state is None:
        state = TranslatorState()
    literals = dict.fromkeys(re.findall(r"'([^']*)'", input_string))
    name_by_literal = {lit: state.register_string(lit) for lit in literals}
    return re.sub(
        r"'([^']*)'",
        lambda m: name_by_literal[m.group(1)],
        input_string,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 7. INHERITANCE HIERARCHY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def is_child(child: str, parent: str, inherits_from: dict) -> bool:
    """Returns ``True`` if *child* is a descendant of *parent* in the hierarchy."""
    if parent in inherits_from[child]:
        return True
    if inherits_from[child] == "_" or inherits_from[child] == ["_"]:
        return False
    return any(is_child(p, parent, inherits_from) for p in inherits_from[child])


def subtypes(subject_class: str, inherits_from: dict) -> list[str]:
    """Return the list of direct and indirect subtypes of *subject_class*."""
    return [e for e in inherits_from if is_child(e, subject_class, inherits_from)]


# ══════════════════════════════════════════════════════════════════════════════
# 8. AST → ALLOY (visitor)
# ══════════════════════════════════════════════════════════════════════════════

# ── 8a. Simple nodes ──────────────────────────────────────────────────────────

def _translate_simple(node) -> str:
    if isinstance(node, BLiteral):
        return "isTrue[True]" if node.val == "True" else "isFalse[False]"
    if isinstance(node, Literal):
        return node.val
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Enumeration):
        return node.name
    if isinstance(node, Nav):
        return ".".join(node.parts)
    raise TypeError(f"Unrecognized node in _simple_translate: {type(node)}")


# ── 8b. if-then-else ─────────────────────────────────────────────────────────

def _translate_ifthenelse(node: IfThenElse, inherits_from: dict, state: TranslatorState) -> str:
    cond = ast_to_alloy(node.cond, inherits_from, state)
    then_part = ast_to_alloy(node.then_expr, inherits_from, state)
    else_part = ast_to_alloy(node.else_expr, inherits_from, state)
    return (
        f"(({cond}) implies ({then_part})) "
        f"and ((not ({cond})) implies ({else_part}))"
    )


# ── 8c. Binary operator ───────────────────────────────────────────────────────

_MAP_OPS: dict[str, str] = {"<>": "!=", "!=": "!=", "and": "&&", "or": "||", "implies": "=>"}


def _normalize_boolean(val: str, op: str) -> str:
    if op in {"=", "!="} and val == "isTrue[True]":
        return "True"
    if op in {"=", "!="} and val == "isFalse[False]":
        return "False"
    return val


def _translate_binaryop_date(
    op: str,
    left: str,
    right: str,
    is_left_lit: bool,
    is_right_lit: bool,
    state: TranslatorState,
) -> str:
    """Translates a comparison that involves at least one date operand.

    *left*/*right* may be date literals (declared as ``one sig`` and replaced
    by their ``dMMDDYYYY`` id) or date-typed attributes (kept as-is).  Ordered
    comparisons dispatch through the registry's registered Alloy code, which
    lives in ``date.als`` (``open date``) — the generator writes that module
    whenever the model has date-typed attributes or date literals.
    """
    extra = ""
    if is_left_lit:
        extra += parse_date(left, state)
    if is_right_lit:
        extra += parse_date(right, state)
    state.write_aux_pred(extra)

    left_val = is_date(left) if is_left_lit else left
    right_val = is_date(right) if is_right_lit else right

    if op in ("=", "!="):
        return f"({left_val} {op} {right_val})"
    return state.date_ops.translate(op, left_val, right_val) or ""


def _translate_binaryop(node: BinaryOp, inherits_from: dict, state: TranslatorState) -> str:
    op = _MAP_OPS.get(node.op, node.op)
    left = _normalize_boolean(ast_to_alloy(node.left, inherits_from, state), op)
    right = _normalize_boolean(ast_to_alloy(node.right, inherits_from, state), op)

    left_lit = is_date(left)
    right_lit = is_date(right)
    is_left_date = bool(left_lit) or _is_date_field(left, state.data)
    is_right_date = bool(right_lit) or _is_date_field(right, state.data)

    if (
        op in (">", ">=", "<", "<=", "=", "!=")
        and (is_left_date and is_right_date)
        and "null" not in (left, right)
    ):
        return _translate_binaryop_date(op, left, right, bool(left_lit), bool(right_lit), state)

    is_left_str = _is_string_operand(left, state)
    is_right_str = _is_string_operand(right, state)

    if (
        op in ("=", "!=")
        and (is_left_str and is_right_str)
        and "null" not in (left, right)
    ):
        str_result = state.string_ops.translate_binary(op, left, right)
        if str_result is None and op == "!=":
            str_result = state.string_ops.translate_binary("<>", left, right)
        if str_result is not None:
            return str_result

    if op == "-":
        return left if right.lower() == "null" else f"minus[{left}, {right}]"

    if op == "+":
        return f"add[{left}, {right}]"

    if op == "!=" and right == "null":
        return f"(! no({left}))"
    if op == "!=" and left == "null":
        return f"(! no({right}))"
    if op == "=" and right == "null":
        return f"( no({left}))"
    if op == "=" and left == "null":
        return f"( no({right}))"

    return f"({left} {op} {right})"


# ── 8d. Calls ─────────────────────────────────────────────────────────────────

def _as_set(expr: str, state: TranslatorState) -> str:
    """Return ``expr`` as a flat Alloy set and record that representation."""
    if not state.is_set_origin:
        state.is_set_origin = True
        return f"image[{expr}]"
    return expr


def _apply_asset(expr: str, state: TranslatorState) -> str:
    """
    Translates ``->asSet()`` to Alloy.

    When the source is a relation (``univ -> univ``), wraps in ``image[...]``
    to obtain the flat set of values.  Either way, marks the result as a set
    so that any subsequent chained operation (``->size()``, ``->union()``, etc.)
    treats it as ``set univ`` instead of ``univ -> univ``.
    """
    return _as_set(expr, state)


def _build_call_handlers() -> dict[str, callable]:
    """
    Builds the handler dict for OCL collection operations.

    Each handler has signature: ``(expr: str, args: list[str], state) -> str``.
    """
    return {
        "size":           lambda expr, args, state: f"#({expr})",
        "excluding":      lambda expr, args, state: (
            f"({_as_set(expr, state)})" if args[0].lower() == "null"
            else f"({_as_set(expr, state)} - {args[0]})"
        ),
        "including":      lambda expr, args, state: (
            f"({_as_set(expr, state)})" if args[0].lower() == "null"
            else f"({_as_set(expr, state)} + {args[0]})"
        ),
        "union":          lambda expr, args, state: f"({_as_set(expr, state)} + {args[0]})",
        "intersection":   lambda expr, args, state: f"({_as_set(expr, state)} & {args[0]})",
        "isempty":        lambda expr, args, state: f"(#({expr}) = 0)",
        "notempty":       lambda expr, args, state: f"(#({expr}) > 0)",
        "closure":        lambda expr, args, state: f"{expr}.*{args[0]}",
        "oclisundefined": lambda expr, args, state: f"no ({expr})",
        "includes":       lambda expr, args, state: (
            f"{args[0]} in image[{expr}]" if not state.is_set_origin else f"{args[0]} in {expr}"
        ),
        "excludes":       lambda expr, args, state: (
            f"not({args[0]} in image[{expr}])" if not state.is_set_origin
            else f"not({args[0]} in {expr})"
        ),
        "includesall":    lambda expr, args, state: (
            f"({args[0]} in image[{expr}])" if not state.is_set_origin else f"({args[0]} in {expr})"
        ),
        "excludesall":    lambda expr, args, state: (
            f"no ({args[0]} & image[{expr}])" if not state.is_set_origin
            else f"no ({args[0]} & {expr})"
        ),
        "asset":          lambda expr, args, state: _apply_asset(expr, state),
    }


CALL_HANDLERS: dict[str, callable] = _build_call_handlers()


def _translate_call(node: Call, inherits_from: dict, estado: TranslatorState) -> str:
    name = node.callname.lower()
    expr = ast_to_alloy(node.expr, inherits_from, estado)
    args = [ast_to_alloy(a, inherits_from, estado) for a in node.args]

    if name == "oclistypeof":
        s = f"{expr} in {args[0]}"
        subs = subtypes(args[0], inherits_from)
        if subs:
            s += " and " + " and ".join(f"{expr} not in {st}" for st in subs)
        return s

    if name == "ocliskindof":
        return f"{expr} in {args[0]}"

    if _field_type_of(expr, estado.data) == "str" or (
        isinstance(node.expr, Call)
        and node.expr.callname.lower() != "size"
        and node.expr.callname.lower() in estado.string_ops.registered_names()
    ):
        translated = estado.string_ops.translate(name, expr, args)
        if translated is not None:
            return translated
        raise StringOpError(
            f"OCL String operation '{node.callname}()' is not supported on a "
            f"String attribute. Registered operations: "
            f"{', '.join(estado.string_ops.registered_names())}."
        )

    handler = CALL_HANDLERS.get(name)
    if handler:
        return handler(expr, args, estado)

    return f"{expr}.{node.callname}"  # fallback


# ── 8e. Iterators ─────────────────────────────────────────────────────────────

def _translate_forall_exists(
    node: IteratorOp,
    coll: str,
    expr: str,
    state: TranslatorState,
) -> str:
    vars_str = ", ".join(node.varnames or ["x"])
    keyword = "all" if node.kind.lower() == "forall" else "some"
    coleccion = coll if state.is_set_origin else f"image[{coll}]"
    return f"{keyword} {vars_str} : {coleccion} | {expr}"


def _translate_select_reject(
    node: IteratorOp,
    coll: str,
    inherits_from: dict,
    state: TranslatorState,
) -> str:
    var = node.varnames[0] if node.varnames else "x"
    es_reject = node.kind.lower() == "reject"

    if not node.generated:
        node.select_id = state.cont_select
        state.cont_select += 1

    id_sel = node.select_id
    neg_open = "not(" if es_reject else ""
    neg_close = ")" if es_reject else ""

    if state.is_set_origin:
        template = (
            f"fun select{id_sel}[suva:set univ]: set univ "
            f"{{{{ ___x___:univ | ___x___ in suva and {neg_open}<expre_select>{neg_close} }}}}"
        )
    else:
        template = (
            f"fun select{id_sel}[suva:univ->univ]: univ->univ "
            f"{{{{ a,___x___:univ | (a->___x___) in suva and {neg_open}<expre_select>{neg_close} }}}}"
        )
        state.is_set_origin = False

    expr_body = ast_to_alloy(node.expr, inherits_from, state).replace(var + ".", "___x___.")
    s = template.replace("<expre_select>", expr_body)

    if not node.generated:
        state.write_aux_pred("\n" + s)
        node.generated = True

    return f"{{select{id_sel}[{coll}]}}"


def _translate_collect(node: IteratorOp, coll: str, expr: str) -> str:
    var = node.varnames[0] if node.varnames else "x"
    e = expr.replace(var + ".", "", 1)
    col = coll.replace(".", ",", 1)
    if "," in col:
        return f" collect[toSeq[{col}],{e}]"
    return f" toSeq[{col},{e}]"


def _translate_iteratorop(node: IteratorOp, inherits_from: dict, state: TranslatorState) -> str:
    coll = ast_to_alloy(node.collection, inherits_from, state)
    expr = ast_to_alloy(node.expr, inherits_from, state)
    kind = node.kind.lower()

    if kind in ("forall", "exists"):
        return _translate_forall_exists(node, coll, expr, state)
    if kind in ("select", "reject"):
        return _translate_select_reject(node, coll, inherits_from, state)
    if kind == "collect":
        state.is_set_origin = False
        return _translate_collect(node, coll, expr)

    return f"{node.kind}({coll})"


def _translate_unaryop(node: UnaryOp, inherits_from: dict, state: TranslatorState) -> str:
    inner = ast_to_alloy(node.operand, inherits_from, state)
    if node.op == "not":
        return f"not ({inner})"
    if node.op == "-":
        return f"minus[0, {inner}]"
    raise ValueError(f"Unrecognized unary operator: {node.op}")


# ── 8f. Main dispatcher ───────────────────────────────────────────────────────

def ast_to_alloy(node, inherits_from: dict, state: TranslatorState) -> str:
    """Visits *node* and returns the equivalent Alloy string."""
    if isinstance(node, (BLiteral, Literal, Var, Enumeration, Nav)):
        return _translate_simple(node)
    if isinstance(node, IfThenElse):
        return _translate_ifthenelse(node, inherits_from, state)
    if isinstance(node, Call):
        return _translate_call(node, inherits_from, state)
    if isinstance(node, BinaryOp):
        return _translate_binaryop(node, inherits_from, state)
    if isinstance(node, UnaryOp):
        return _translate_unaryop(node, inherits_from, state)
    if isinstance(node, IteratorOp):
        return _translate_iteratorop(node, inherits_from, state)
    raise TypeError(f"Unknown node type: {type(node)}")


# ── Intermediate pipeline ─────────────────────────────────────────────────────

def predicate_tokens_to_str(
    tokens: list[Token],
    inherits_from: dict,
    state: TranslatorState,
) -> str:
    """Converts a token list directly to Alloy (parse + codegen)."""
    if not tokens:
        return ""
    ast = parse_predicate(tokens)
    return ast_to_alloy(ast, inherits_from, state)


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def ocl_to_alloy(
    inherits_from: dict,
    data: dict,
    ocl: str,
    context_name: str = "",
    state: TranslatorState | None = None,
    enums: dict[str, set[str]] | None = None,
) -> str:
    """Translates an OCL expression to an Alloy fact.

    Args:
        inherits_from: ``{ClassName: [Parent, ...] | '_'}`` hierarchy map.
        data:          ``{ClassName: ['field:Type', ...]}`` attribute map.
        ocl:           OCL expression as a string.
        context_name:  Name of the class the constraint applies to.
        state:        Shared state between constraints (created if ``None``).
        enums:         ``{EnumName: {Literal, ...}}`` mapping used to
                       validate ``EnumType::Literal`` references before
                       translation. ``None`` (default) skips the check.

    Returns:
        A string with the generated Alloy fact (including auxiliary predicates
        and string/date sigs when applicable).

    Raises:
        EnumReferenceError: If *enums* is provided and the OCL references an
            unknown enumeration type or an undeclared literal.
    """
    if state is None:
        state = TranslatorState()
    state.set_enums(enums)
    state.data = data
    state.init_constraint()

    tree, parser = parse_ocl_expression(ocl)
    toks = tokenize_tree(tree, parser)
    validate_enum_references(toks, state, context_name)
    toks = write_prefix_ocl(toks, data, inherits_from, context_name)
    invariante = predicate_tokens_to_str(toks, inherits_from, state)
    pred_aux = state.read_aux_pred()
    result = pred_aux + f"fact{{ all self:this/{context_name}|{invariante}}}"
    return process_string_types(result, state)
