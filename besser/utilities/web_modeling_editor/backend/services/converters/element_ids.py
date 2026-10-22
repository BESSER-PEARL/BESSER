"""Element ids: which ids the editor can use, and carrying ids over to a rebuilt model."""

import logging
import re

from besser.BUML.metamodel.structural import DomainModel, Enumeration

logger = logging.getLogger(__name__)

# Editor ids are 36-character uuids; the cap stops a hostile id from bloating every reference to it.
MAX_ELEMENT_ID_LENGTH = 128

# The editor keys plain objects by id and the GUI parser splits a field reference at '.', so
# only these characters are kept ('__proto__' matches but would hit the object prototype).
_ELEMENT_ID_PATTERN = re.compile(rf"[A-Za-z0-9_:-]{{1,{MAX_ELEMENT_ID_LENGTH}}}")


def is_valid_element_id(value) -> bool:
    """True when ``value`` can be used as an editor element id."""
    return isinstance(value, str) and bool(_ELEMENT_ID_PATTERN.fullmatch(value)) and value != "__proto__"


def adopt_element_ids(model: DomainModel, previous: DomainModel) -> None:
    """Give each id-less element of ``model`` the id of its counterpart in ``previous``.

    For a model rebuilt on top of an existing one (an image merged into a
    diagram), so the elements that were already there keep the ids other
    diagrams refer to. Counterparts are matched by id when both carry one,
    otherwise by case-insensitive name within the same parent (types by kind
    and name, members within their class or enumeration, generalizations
    within their specific class). An id is never given twice, and renames
    are not guessed.
    """
    used = {element.id for element in _all_elements(model) if element.id}

    def adopt(new, old):
        if new.id is None and old.id is not None and old.id not in used:
            new.id = old.id
            used.add(old.id)

    for new_type, old_type in _counterparts(model.types, previous.types, _type_key):
        adopt(new_type, old_type)
        for members in ("attributes", "methods", "literals"):
            for new, old in _counterparts(getattr(new_type, members, ()), getattr(old_type, members, ()), _name_key):
                adopt(new, old)
    for new, old in _counterparts(model.associations, previous.associations, _name_key):
        adopt(new, old)
    for new, old in _counterparts(model.generalizations, previous.generalizations, _generalization_key):
        adopt(new, old)


def _name_key(element):
    return (element.name or "").lower()


def _type_key(element):
    return isinstance(element, Enumeration), _name_key(element)


def _generalization_key(generalization):
    return _name_key(generalization.specific), _name_key(generalization.general)


def _counterparts(new_elements, old_elements, key):
    """Pairs ``(new, old)``: same id when the new element has one, else the same key.

    A key shared by two old elements matches neither.
    """
    old_by_id = {old.id: old for old in old_elements if old.id}
    old_by_key, ambiguous = {}, set()
    for old in old_elements:
        k = key(old)
        if k in old_by_key:
            ambiguous.add(k)
        old_by_key[k] = old
    for new in new_elements:
        if new.id is not None:
            old = old_by_id.get(new.id)
        elif key(new) in ambiguous:
            logger.warning("Not reusing an element id for %s %r: the previous model has several matches.",
                           type(new).__name__, getattr(new, "name", ""))
            old = None
        else:
            old = old_by_key.get(key(new))
        if old is not None:
            yield new, old


def _all_elements(model: DomainModel):
    for model_type in model.types:
        yield model_type
        for members in ("attributes", "methods", "literals"):
            yield from getattr(model_type, members, ())
    yield from model.associations
    yield from model.generalizations
