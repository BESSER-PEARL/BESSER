"""Bottom-up resolution of anonymous class / data expressions, plus the
union/intersection materialisation rule.

Every rule that touches an anonymous expression (restrictions, union domains,
nested ``allValuesFrom``) goes through :class:`ExpressionResolver`, so the
"innermost first" behaviour and aux-class de-duplication live in one place.

A materialised class owns the feature it constrains — an object restriction
links the property to its auxiliary class (``add_association``), and a data
restriction declares it there (``add_attribute``). Without that, the invariant
the auxiliary class carries names a feature declared only on its subclasses, and
OCL resolves ``self.<p>`` by walking a context's ancestors, never its subclasses.

The resolver mutates the :class:`~owl2uml.model.UMLModel` owned by the
``Mapper`` it is bound to, via the mapper's ``ensure_*`` / ``add_*`` helpers.
Aux classes are de-duplicated by content *signature*.
``min 1 hasPart`` restrictions collapse to one aux class that every subsuming
class generalises to.
"""
from __future__ import annotations

from rdflib import URIRef, BNode, Literal
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.collection import Collection as RDFList

from . import naming

_CARD = {
    "min": (OWL.minCardinality, OWL.minQualifiedCardinality),
    "max": (OWL.maxCardinality, OWL.maxQualifiedCardinality),
    "exact": (OWL.cardinality, OWL.qualifiedCardinality),
}
_CARD_MULT = {"min": lambda k: (k, "*"), "max": lambda k: (0, k), "exact": lambda k: (k, k)}
_CARD_OP = {"min": ">=", "max": "<=", "exact": "="}
_CARD_OBJ_RULE = {"min": "D23", "max": "D24", "exact": "D25"}
_CARD_DATA_RULE = {"min": "O13", "max": "O14", "exact": "O15"}


class ExpressionResolver:
    def __init__(self, mapper):
        self.m = mapper
        self.g = mapper.g
        self.ax = mapper.ax
        self._memo: dict = {}          # signature -> aux class name

    def _aux(self, sig, base_name: str, abstract: bool = False):
        """Return ``(name, is_new)`` for an aux class identified by ``sig``."""
        if sig in self._memo:
            return self._memo[sig], False
        name = self.m.make_aux_class(base_name, abstract=abstract)
        self._memo[sig] = name
        return name, True

    # ---- class expressions ----------------------------------------------
    def resolve_class(self, node, subsuming: str | None = None) -> str:
        if isinstance(node, URIRef):
            name = self.m.ensure_class(node)
            return self._finish(name, subsuming)
        if (node, OWL.unionOf, None) in self.g:
            return self._finish(self._union(node), subsuming)
        if (node, OWL.intersectionOf, None) in self.g:
            return self._finish(self._intersection(node), subsuming)
        if (node, OWL.oneOf, None) in self.g:
            return self._finish(self._one_of(node), subsuming)
        if (node, OWL.complementOf, None) in self.g:
            return self._finish(self._complement(node), subsuming)
        if (node, RDF.type, OWL.Restriction) in self.g:
            return self._finish(self._restriction(node), subsuming)
        return self.m.ensure_class(OWL.Thing)

    def _finish(self, name: str, subsuming: str | None) -> str:
        # only wire a generalization to a *class* (a oneOf may resolve to an enum)
        if subsuming and subsuming != name and name in self.m.model.classes:
            self.m.add_generalization(subsuming, name)
        return name

    # ---- union / intersection / oneOf / complement ----------------------
    def _list(self, node, pred):
        lst = self.g.value(node, pred)
        return list(RDFList(self.g, lst)) if lst is not None else []

    def _union(self, node) -> str:
        ops = sorted(dict.fromkeys(self.resolve_class(i) for i in self._list(node, OWL.unionOf)))
        if len(ops) == 1:
            return ops[0]
        member = self._subsuming_member(ops)
        if member:
            return member
        name, is_new = self._aux(("union", tuple(ops)), naming.union_name(ops), abstract=True)
        if is_new:
            for op in ops:
                self.m.add_generalization(op, name)       # Ci  ▷  _Union  (D19)
        return name

    def _intersection(self, node) -> str:
        ops = sorted(dict.fromkeys(self.resolve_class(i) for i in self._list(node, OWL.intersectionOf)))
        if len(ops) == 1:
            return ops[0]
        name, is_new = self._aux(("inter", tuple(ops)), naming.intersection_name(ops))
        if is_new:
            for op in ops:
                self.m.add_generalization(name, op)       # _Intersection ▷ Ci  (D18)
        return name

    def _one_of(self, node) -> str:
        items = self._list(node, OWL.oneOf)
        names = [naming.sanitize(naming.local_name(i)) for i in items]
        if items and all(isinstance(i, Literal) for i in items):
            return self.m.make_enum([self.m.literal_value(i) for i in items],
                                    hint=naming.one_of_name(names))            # D17
        if any(self._is_relational(i) for i in items):                         # D21
            name, is_new = self._aux(("oneof", tuple(sorted(map(str, items)))),
                                     naming.one_of_name(names))
            if is_new:
                for i in items:
                    self.m.ensure_instance(i)
            return name
        return self.m.make_enum(names, hint=naming.one_of_name(names))         # D20

    def _complement(self, node) -> str:
        inner = self.resolve_class(self.g.value(node, OWL.complementOf))
        name, is_new = self._aux(("not", inner), naming.not_name(inner), abstract=True)
        if is_new:
            self.m.add_ocl(name, f"not self.oclIsKindOf({inner})", origin="O06")     # O06
        return name

    # ---- subsuming-member shortcut for domain/range unions --------------
    def _subsuming_member(self, ops: list[str]) -> str | None:
        """Return a union member that is a (transitive) superclass of every
        other member, if one exists.

        The union of a class with its own subclasses is just that class, so
        collapsing to such a member is lossless. Collapsing to an *external*
        common ancestor (e.g. every member being an ``rdfs:subClassOf`` of a
        broad class like ``Thing``) is not — it would widen the domain to that
        ancestor's other subclasses — so those unions are materialised instead.
        """
        for cand in ops:
            if all(cand in self._ancestors(o) for o in ops if o != cand):
                return cand
        return None

    def _ancestors(self, name: str) -> set[str]:
        iri = self.m.iri_for_name(name)
        if iri is None:
            return set()
        return {self.m.ensure_class(s)
                for s in self.g.transitive_objects(iri, RDFS.subClassOf)
                if isinstance(s, URIRef) and s != iri}

    # ---- restrictions (D22-D25, O07, O08, O09-O15) ----------------------
    def _restriction(self, node) -> str:
        prop = self.g.value(node, OWL.onProperty)
        if prop is None:
            return self.m.ensure_class(OWL.Thing)
        p = self.m.canon.name(prop)
        is_obj = self.ax.is_object_property(prop) or not self.ax.is_data_property(prop)

        val = self.g.value(node, OWL.someValuesFrom)
        if val is not None:
            return self._some(p, val, is_obj)
        val = self.g.value(node, OWL.allValuesFrom)
        if val is not None:
            return self._all(p, val, is_obj)
        val = self.g.value(node, OWL.hasValue)
        if val is not None:
            return self._has_value(p, val, is_obj)
        if self.g.value(node, OWL.hasSelf) is not None:
            return self._has_self(p)
        for kind, preds in _CARD.items():
            n = self.g.value(node, preds[0])
            if n is None:
                n = self.g.value(node, preds[1])
            if n is not None:
                return self._cardinality(node, prop, p, is_obj, kind, int(n))
        return self.m.ensure_class(OWL.Thing)

    def _some(self, p, val, is_obj) -> str:            # D22 / O10
        if is_obj:
            d = self.resolve_class(val)
            name, new = self._aux(("some", p, d, True), naming.restriction_name("some", p, d))
            if new:
                self.m.add_association(name, d, p, (0, "*"), (1, "*"))
        else:
            dr = self.m.resolve_data_range(val)
            name, new = self._aux(("some", p, dr, False), naming.restriction_name("some", p, dr))
            if new:
                self.m.add_attribute(name, p, dr, (1, "*"))
                self.m.add_ocl(name, f"self.{p}->asSet()->exists(v | {self.m.satisfies(dr, 'v')})", origin="O10")
        return name

    def _all(self, p, val, is_obj) -> str:             # O07 / O11
        if is_obj:
            d = self.resolve_class(val)
            name, new = self._aux(("all", p, d, True), naming.restriction_name("all", p, d))
            if new:
                self.m.add_association(name, d, p, (0, "*"), (0, "*"))
                self.m.add_ocl(name, f"self.{p}->forAll(o | o.oclIsKindOf({d}))", origin="O07")
        else:
            dr = self.m.resolve_data_range(val)
            name, new = self._aux(("all", p, dr, False), naming.restriction_name("all", p, dr))
            if new:
                self.m.add_attribute(name, p, dr, (0, "*"))
                self.m.add_ocl(name, f"self.{p}->asSet()->forAll(v | {self.m.satisfies(dr, 'v')})", origin="O11")
        return name

    def _has_value(self, p, val, is_obj) -> str:       # O08 / O12
        if is_obj and isinstance(val, (URIRef, BNode)):
            a = self.m.ensure_instance(val)
            ta = self.m.classifier_name(val)
            name, new = self._aux(("hasValue", p, str(val), True), naming.restriction_name("hasValue", p, a))
            if new:
                self.m.add_association(name, ta, p, (0, "*"), (1, "*"))
                self.m.add_ocl(name, f"self.{p}->includes({a})", origin="O08")
        else:
            lit = self.m.literal_value(val)
            name, new = self._aux(("hasValue", p, str(val), False),
                                  naming.restriction_name("hasValue", p, naming.sanitize(str(val))))
            if new:
                self.m.add_attribute(name, p, self.m.resolve_data_range(None), (1, "*"))
                self.m.add_ocl(name, f"self.{p}->asSet()->includes({lit})", origin="O12")
        return name

    def _has_self(self, p) -> str:                     # O09
        name, new = self._aux(("hasSelf", p), naming.restriction_name("hasSelf", p, ""))
        if new:
            self.m.add_association(name, name, p, (0, "*"), (0, "*"))
            self.m.add_ocl(name, f"self.{p}->includes(self)", origin="O09")
        return name

    def _cardinality(self, node, prop, p, is_obj, kind, k) -> str:   # D23-25 / O13-15
        if is_obj:
            filler = self.g.value(node, OWL.onClass)
            d = self.resolve_class(filler) if filler is not None else self._range_filler(prop)
            name, new = self._aux((kind, p, d, k, True), naming.restriction_name(kind, p, d, n=k))
            if new:
                self.m.add_association(name, d, p, (0, "*"), _CARD_MULT[kind](k))
        else:
            filler = self.g.value(node, OWL.onDataRange)
            # No onDataRange: fall back to the widest primitive. Resolved through
            # the mapper so the name is a real BUML type, not the literal
            # "String" the reference hardcodes here.
            dr = (self.m.resolve_data_range(filler) if filler is not None
                  else self.m.resolve_data_range(None))
            name, new = self._aux((kind, p, dr, k, False), naming.restriction_name(kind, p, dr, n=k))
            if new:
                self.m.add_attribute(name, p, dr, _CARD_MULT[kind](k))
                self.m.add_ocl(
                    name,
                    f"self.{p}->asSet()->select(v | {self.m.satisfies(dr, 'v')})->size() {_CARD_OP[kind]} {k}",
                    origin=_CARD_DATA_RULE[kind],
                )
        return name

    def _range_filler(self, prop) -> str:
        rng = self.ax.ranges(prop)
        return self.resolve_class(rng[0]) if rng else self.m.ensure_class(OWL.Thing)

    def _is_relational(self, ind) -> bool:
        return any(p in self.ax.object_props for p, _ in self.g.predicate_objects(ind))
