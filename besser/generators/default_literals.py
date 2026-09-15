"""Render a modelled ``default_value`` as a safe Python literal.

A template that interpolates ``default_value`` directly writes whatever the
model carries into executable source. That is a code-injection sink, because
``default_value`` reaches the metamodel unvalidated from request JSON
(``class_diagram_processor`` passes ``attr.get("defaultValue")`` straight to
``Property``), and ``Property.default_value``'s setter is a bare assignment.

Observed 2026-09-14 on the SQLAlchemy template, which emitted the value
UNQUOTED for any non-``str``/``bool``/enum type::

    pages: Mapped_[int] = mapped_column(Integer_, default=__import__("os").getcwd())

``SQLGenerator`` then executes the generated module in a subprocess
(``sql_generator.py`` runs ``[sys.executable, temp_py_path]``) to dump DDL, so
that expression runs as the backend user. The ``str`` branch was injectable
too — the value sat inside a ``"`` it could close.

``python_default`` closes the sink at the point of rendering: the value is
coerced to the attribute's declared type and emitted through ``repr()``, so the
result is always a literal and never an expression. A value that cannot be
coerced raises at generation time rather than producing runnable source.
"""

from __future__ import annotations

import keyword
from typing import Any


class InvalidDefaultValueError(ValueError):
    """A modelled default cannot be expressed as a literal of its type."""


_TRUE = {"true", "yes", "1", "on", "t", "y"}
_FALSE = {"false", "no", "0", "off", "f", "n"}


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise InvalidDefaultValueError(f"{raw!r} is not a boolean")


def python_default(raw: Any, type_name: str, *, owner: str = "") -> str:
    """Return *raw* as a Python literal appropriate for ``type_name``.

    ``owner`` is only used to make the error message locatable.

    Raises:
        InvalidDefaultValueError: when the value is not expressible as a
            literal of that type — which is also what a code-injection attempt
            looks like.
    """
    where = f" for {owner}" if owner else ""
    name = (type_name or "").lower()

    try:
        if name in ("str", "string"):
            return repr(str(raw))
        if name in ("bool", "boolean"):
            return repr(_as_bool(raw))
        if name in ("int", "integer"):
            return repr(int(str(raw).strip()))
        if name in ("float", "double", "decimal"):
            return repr(float(str(raw).strip()))
    except InvalidDefaultValueError:
        raise
    except (TypeError, ValueError) as exc:
        raise InvalidDefaultValueError(
            f"default value {raw!r}{where} is not a valid {type_name}: {exc}"
        ) from None

    # Any other declared type (date, datetime, time, custom): only a plain
    # literal is safe to emit. Reject anything that is not one rather than
    # writing it into source.
    text = str(raw).strip()
    if not text:
        raise InvalidDefaultValueError(f"empty default value{where}")
    return repr(text)


def enum_default(raw: Any, enum_name: str, *, members: Any = None,
                 owner: str = "") -> str:
    """Return ``EnumName.MEMBER`` after checking MEMBER is a bare identifier.

    Without the identifier check the member name is another injection point:
    the template emits ``{{ type.name }}.{{ default_value }}`` verbatim.
    """
    where = f" for {owner}" if owner else ""
    member = str(raw).strip()
    # Callers disagree about whether the model stores the member qualified.
    # The pydantic template, for one, accepts both "Status.OPEN" and "OPEN".
    # Accept the qualified form, but only with the enum's own name.
    prefix = f"{enum_name}."
    if member.startswith(prefix):
        member = member[len(prefix):].strip()
    if not member.isidentifier() or keyword.iskeyword(member):
        raise InvalidDefaultValueError(
            f"enum default {raw!r}{where} is not a valid member name"
        )
    if members is not None:
        known = {str(getattr(m, "name", m)) for m in members}
        if known and member not in known:
            raise InvalidDefaultValueError(
                f"enum default {member!r}{where} is not a member of "
                f"{enum_name} (known: {', '.join(sorted(known))})"
            )
    return f"{enum_name}.{member}"


def docstring_default(raw: Any, type_name: str, *, owner: str = "") -> str:
    """Render a modelled default for display inside a triple-quoted docstring.

    ``python_default`` is not enough here. It returns a Python literal, and the
    repr of a string containing a triple-double-quote still *contains* that
    sequence -- which closes the surrounding docstring and drops whatever
    follows into the function body at statement indentation.

    Observed on ``backend/templates/router.py.j2``, whose generated routers are
    executed by ``/besser_api/deploy-app``: the value line was hardened while
    the two docstring lines a few rows above still interpolated raw. That was
    the third site of the same sink found in this branch.

    So: coerce through ``python_default`` first (which rejects anything not
    expressible as a literal of the declared type), then make the result
    docstring-safe -- no double quote survives to form the closing sequence,
    and no newline reaches column 0.
    """
    literal = python_default(raw, type_name, owner=owner)
    flattened = " ".join(literal.splitlines())
    return flattened.replace('"', "'")


def register_default_literals(env) -> None:
    """Expose ``python_default`` / ``enum_default`` to a Jinja environment.

    Every template that renders a modelled default must go through these.
    Registering from one place keeps a new generator from quietly reopening
    the sink by interpolating ``default_value`` directly.
    """
    env.filters["python_default"] = python_default
    env.filters["docstring_default"] = docstring_default
    env.globals.update(python_default=python_default, enum_default=enum_default,
                       docstring_default=docstring_default)
