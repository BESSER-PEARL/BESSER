"""SHACL shapes -> OCL invariants (Table ``tab:shacl-ocl`` of the KG2UML spec).

This is phase 2 of the framework: it runs as the final :class:`~owl2uml.mapping.Mapper`
phase when a shapes graph is supplied, *after* the OWL-2 -> UML/OCL conversion. It does
not create new structure; it attaches additional OCL invariants to the classes the OWL
phase already produced.

It reuses the Mapper's :class:`~owl2uml.canonicalize.Canonicalizer` — so a SHACL
``sh:path`` / ``sh:targetClass`` / ``sh:node`` IRI resolves to the *same* UML name the
OWL phase used — plus its ``ensure_class`` / ``add_ocl`` / ``resolve_data_range`` /
``literal_value`` helpers and its model.

Each constraint on a property shape becomes its own named OCL invariant attached to the
shape's target class. Constraints with no OCL equivalent (``sh:nodeKind``, ``sh:closed``,
``sh:flags``, ``sh:languageIn``, ``sh:severity``, ``sh:message``) are skipped.
"""
from __future__ import annotations

from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import RDF
from rdflib.collection import Collection as RDFList

SH = Namespace("http://www.w3.org/ns/shacl#")


def _regex_literal(s: str) -> str:
    """OCL single-quoted string literal for a ``sh:pattern`` regex.

    Backslashes are left literal (not doubled) so ``^\\d{9}$`` renders as
    ``'^\\d{9}$'`` — regex fidelity matters more than OCL string-escape purity here.
    """
    return "'" + s.replace("'", "\\'").replace("\n", " ") + "'"


class ShaclMapper:
    def __init__(self, mapper, shapes):
        self.m = mapper
        self.g = shapes                       # the SHACL graph (separate from the OWL graph)
        self.canon = mapper.canon             # OWL-built canonicalizer (shared)
        self._seen: set[tuple[str, str]] = set()   # (context, body) dedup
        # predicate -> handler(property_shape, path_name) -> list[(ocl_body, suffix)]
        self.rules = {
            SH.datatype:         self._c_datatype,
            SH.node:             self._c_node,
            SH["class"]:         self._c_class,
            SH["or"]:            lambda ps, p: self._c_logic(ps, p, SH["or"], " or "),
            SH["and"]:           lambda ps, p: self._c_logic(ps, p, SH["and"], " and "),
            SH["not"]:           self._c_not,
            SH["in"]:            self._c_in,
            SH.hasValue:         self._c_hasvalue,
            SH.pattern:          self._c_pattern,
            SH.uniqueLang:       self._c_uniquelang,
            SH.minCount:         lambda ps, p: self._c_count(ps, p, SH.minCount, ">="),
            SH.maxCount:         lambda ps, p: self._c_count(ps, p, SH.maxCount, "<="),
            SH.minLength:        lambda ps, p: self._c_length(ps, p, SH.minLength, ">="),
            SH.maxLength:        lambda ps, p: self._c_length(ps, p, SH.maxLength, "<="),
            SH.minInclusive:     lambda ps, p: self._c_range(ps, p, SH.minInclusive, ">="),
            SH.maxInclusive:     lambda ps, p: self._c_range(ps, p, SH.maxInclusive, "<="),
            SH.minExclusive:     lambda ps, p: self._c_range(ps, p, SH.minExclusive, ">"),
            SH.maxExclusive:     lambda ps, p: self._c_range(ps, p, SH.maxExclusive, "<"),
            SH.disjoint:         self._c_disjoint,
            SH.equals:           self._c_equals,
            SH.lessThan:         lambda ps, p: self._c_compare(ps, p, SH.lessThan, "<"),
            SH.lessThanOrEquals: lambda ps, p: self._c_compare(ps, p, SH.lessThanOrEquals, "<="),
            SH.xone:             self._c_xone,
            SH.qualifiedValueShape: self._c_qualified,
        }
        # Rows with no OCL equivalent are intentionally absent from self.rules:
        #   sh:nodeKind, sh:closed, sh:flags, sh:languageIn, sh:severity, sh:message

    # ---- orchestration ---------------------------------------------------
    def run(self) -> None:
        # Sorted throughout: invariant *order* within a context class is
        # observable output, and rdflib returns result sets in hash order.
        for shape in sorted(set(self.g.subjects(RDF.type, SH.NodeShape)), key=str):
            targets = self._targets(shape)
            for ctx in targets:
                for ps in sorted(self.g.objects(shape, SH.property), key=str):
                    self._map_property_shape(ctx, ps)

    def _targets(self, shape) -> list[str]:
        """UML context class name(s) for a NodeShape."""
        tcs = sorted((t for t in self.g.objects(shape, SH.targetClass) if isinstance(t, URIRef)), key=str)
        if not tcs and isinstance(shape, URIRef):
            tcs = [shape]                     # implicit class target (shape IRI is a class)
        return [self.m.ensure_class(t) for t in tcs]

    def _map_property_shape(self, ctx: str, ps) -> None:
        path = self.g.value(ps, SH.path)
        if not isinstance(path, URIRef):
            self.m.warn(
                "SHACL_COMPLEX_PATH",
                f"Skipping property shape {ps}: sequence/inverse/alternative "
                f"sh:path expressions have no UML feature to attach to.",
            )
            return
        pname = self.canon.name(path)
        if not self.m._has_property(ctx, pname):
            # sh:path points at a property the OWL phase never materialised as an
            # attribute or association on ctx (or an ancestor) — shapes graphs
            # commonly constrain rdfs:label/rdfs:comment, which are annotation
            # properties and so never become a UML feature. Emitting ``self.<pname>``
            # would
            # reference a feature the
            # context class doesn't have, so the whole property shape is skipped.
            self.m.warn(
                "SHACL_PATH_NOT_MODELLED",
                f"Skipping SHACL constraints on {ctx}.{pname}: the OWL phase "
                f"never materialised that property as an attribute or association.",
            )
            return
        self._ctx = ctx     # stashed for handlers that retarget the owning attribute's type
        for pred in sorted(set(self.g.predicates(ps)), key=str):
            handler = self.rules.get(pred)
            if handler is None:
                continue
            for body, suffix in handler(ps, pname):
                if body:
                    self._add(ctx, body, pname, suffix)

    def _add(self, ctx: str, body: str, pname: str, suffix: str) -> None:
        key = (ctx, body)
        if key in self._seen:
            return
        self._seen.add(key)
        existing = self.m.model.classes.get(ctx)
        if existing is not None and any(i.body == body for i in existing.invariants):
            return                            # identical invariant already present (e.g. from OWL)
        self.m.add_ocl(ctx, body, origin=f"S-{suffix}", name=f"{pname}_{suffix}")

    # ---- node / class / logical resolvers --------------------------------
    def _node_datatype(self, node) -> str | None:
        """Resolve a ``sh:or``/``sh:and``/``sh:not``/``sh:node``/``sh:class``/``sh:xone``
        member that is a *datatype wrapper* (``<node> sh:datatype <xsd:type>``, the
        wrapper idiom shapes graphs use for ``datatype-xsd-*`` resources) to a UML DataType
        name. Reuses :meth:`Mapper.resolve_data_range`, so it is the *same* DataType
        (e.g. the primitive for ``xsd:dateTime``) a plain ``sh:datatype`` constraint
        or the OWL phase would already have produced for that XSD type — this is what
        keeps a shared blank node (e.g. two properties pointing at the same ``sh:or``
        list) from ever fanning out into duplicate types. Returns ``None`` for
        anything that is not such a wrapper (i.e. an actual class reference).
        """
        if node is None:
            return None
        dt = self.g.value(node, SH.datatype)
        return self.m.resolve_data_range(dt) if dt is not None else None

    def _node_class(self, node) -> str | None:
        """Resolve a ``sh:node`` value / ``sh:or`` list member to a UML class name.

        Handles (a) a plain class IRI, (b) a NodeShape carrying ``sh:targetClass``,
        (c) an ``sh-node-*`` wrapper resource carrying ``sh:node <Class>``
        (recurse), and (d) an inline anonymous shape carrying ``sh:class <Class>``
        — the natural way to write ``sh:or ( [ sh:class :Pet ] [ sh:class :Toy ] )``
        and what ``sh:qualifiedValueShape`` almost always looks like. Without (d)
        a blank-node member resolves to nothing and the whole constraint is
        silently skipped.

        A node carrying ``sh:datatype`` is a datatype wrapper (see
        ``_node_datatype``), not a class, and is excluded here so it doesn't get
        materialised as a bogus stub class.
        """
        if node is None or self.g.value(node, SH.datatype) is not None:
            return None
        inner = self.g.value(node, SH.node)          # wrapper / nested node shape (c)
        if inner is not None:
            return self._node_class(inner)
        cls = self.g.value(node, SH["class"])        # inline shape (d)
        if isinstance(cls, URIRef):
            return self.m.ensure_class(cls)
        tc = self.g.value(node, SH.targetClass)      # nested NodeShape (b)
        if isinstance(tc, URIRef):
            return self.m.ensure_class(tc)
        if isinstance(node, URIRef):                 # plain class IRI (a)
            return self.m.ensure_class(node)
        return None

    def _node_check(self, node, var: str = "v") -> str | None:
        """OCL boolean test for one ``sh:node``/``sh:class``/``sh:or``/``sh:and``/
        ``sh:not``/``sh:xone`` member: ``oclIsTypeOf`` for a datatype wrapper,
        ``oclIsKindOf`` for a class. ``None`` if the member resolves to neither.
        """
        dt = self._node_datatype(node)
        if dt:
            return f"{var}.oclIsTypeOf({dt})"
        cls = self._node_class(node)
        return f"{var}.oclIsKindOf({cls})" if cls else None

    def _c_node(self, ps, pname):
        chk = self._node_check(self.g.value(ps, SH.node))
        return [(f"self.{pname}->forAll(v | {chk})", "node")] if chk else []

    def _c_class(self, ps, pname):
        chk = self._node_check(self.g.value(ps, SH["class"]))
        return [(f"self.{pname}->forAll(v | {chk})", "class")] if chk else []

    def _c_logic(self, ps, pname, pred, joiner):
        lst = self.g.value(ps, pred)
        if lst is None:
            return []
        members = list(RDFList(self.g, lst))
        dt_names = [self._node_datatype(e) for e in members]
        if members and all(dt_names):
            # O01/O02: every operand is a datatype (the sh:or-of-xsd-types
            # pattern) -> reuse/materialise the shared composite DataType (same
            # dedup the OWL owl:unionOf/intersectionOf phase uses) and retarget
            # the property's own attribute to it, rather than duplicating an
            # inline "oclIsTypeOf(A) or oclIsTypeOf(B) ..." check on every class
            # that happens to declare this property.
            kind = "union" if pred == SH["or"] else "intersection"
            composite = self.m.composite_datarange(dt_names, kind)
            self.m.set_attribute_type(self._ctx, pname, composite)
            return []                        # the invariant now lives on the DataType itself
        checks = list(dict.fromkeys(c for c in (self._node_check(e) for e in members) if c))
        if not checks:
            return []
        suffix = "or" if pred == SH["or"] else "and"
        return [(f"self.{pname}->forAll(v | {joiner.join(checks)})", suffix)]

    def _c_not(self, ps, pname):
        node = self.g.value(ps, SH["not"])
        dt = self._node_datatype(node)
        if dt:
            # O03: reuse/materialise the shared complement DataType and retarget
            # the property's attribute to it (see _c_logic).
            composite = self.m.composite_datarange([dt], "complement")
            self.m.set_attribute_type(self._ctx, pname, composite)
            return []
        chk = self._node_check(node)
        return [(f"self.{pname}->forAll(v | not {chk})", "not")] if chk else []

    def _c_xone(self, ps, pname):
        lst = self.g.value(ps, SH.xone)
        if lst is None:
            return []
        checks = list(dict.fromkeys(c for c in (self._node_check(e)
                                                 for e in RDFList(self.g, lst)) if c))
        if not checks:
            return []
        # B-OCL has no collection literal, so the sum is folded with `+`
        # instead of `Sequence{...}->sum()`. Same value: the sequence only ever
        # held these indicator terms. Each `if ... endif` is parenthesised so
        # the fold cannot be absorbed into an `else` branch.
        terms = " + ".join(f"(if {c} then 1 else 0 endif)" for c in checks)
        return [(f"self.{pname}->forAll(v | {terms} = 1)", "xone")]

    def _c_qualified(self, ps, pname):
        node = self.g.value(ps, SH.qualifiedValueShape)
        chk = self._node_check(node)
        if not chk:
            return []
        out = []
        qmin = self.g.value(ps, SH.qualifiedMinCount)
        qmax = self.g.value(ps, SH.qualifiedMaxCount)
        sel = f"self.{pname}->select(v | {chk})->size()"
        if qmin is not None:
            out.append((f"{sel} >= {self.m.ocl_literal(qmin)}", "qualifiedMinCount"))
        if qmax is not None:
            out.append((f"{sel} <= {self.m.ocl_literal(qmax)}", "qualifiedMaxCount"))
        return out

    # ---- scalar / literal resolvers --------------------------------------
    def _c_datatype(self, ps, pname):
        t = self.m.resolve_data_range(self.g.value(ps, SH.datatype))
        return [(f"self.{pname}->forAll(v | v.oclIsTypeOf({t}))", "datatype")]

    def _c_count(self, ps, pname, pred, op):
        n = self.m.ocl_literal(self.g.value(ps, pred))
        return [(f"self.{pname}->size() {op} {n}", pred.split("#")[-1])]

    def _c_length(self, ps, pname, pred, op):
        n = self.m.ocl_literal(self.g.value(ps, pred))
        return [(f"self.{pname}->forAll(v | v.size() {op} {n})", pred.split("#")[-1])]

    def _c_range(self, ps, pname, pred, op):
        n = self.m.ocl_literal(self.g.value(ps, pred))
        return [(f"self.{pname}->forAll(v | v {op} {n})", pred.split("#")[-1])]

    def _c_pattern(self, ps, pname):
        return [(f"self.{pname}->forAll(v | v.matches({_regex_literal(str(o))}))", "pattern")
                for o in sorted(self.g.objects(ps, SH.pattern), key=str)]

    def _c_uniquelang(self, ps, pname):
        v = self.g.value(ps, SH.uniqueLang)
        if isinstance(v, Literal) and v.value is True:
            # `isUnique(body)` evaluates `body` per element under the implicit
            # iterator and requires the results to be pairwise distinct — the
            # same thing `collect(p | p.language)->isUnique()` meant, but in the
            # form B-OCL accepts (the zero-argument spelling is not valid OCL).
            return [(f"self.{pname}->isUnique(language)", "uniqueLang")]
        return []

    def _c_hasvalue(self, ps, pname):
        return [(f"self.{pname}->includes({self._ocl_value(o)})", "hasValue")
                for o in sorted(self.g.objects(ps, SH.hasValue), key=str)]

    def _c_in(self, ps, pname):
        lst = self.g.value(ps, SH["in"])
        if lst is None:
            return []
        # An empty `sh:in ()` survives the KG projection as a one-element list
        # holding rdf:nil, so drop the terminator before reading the members.
        members = [self._ocl_value(x) for x in RDFList(self.g, lst) if x != RDF.nil]
        if not members:
            # `Set{}->includes(v)` is false for every v, so the invariant held
            # only when the property had no values at all.
            return [(f"self.{pname}->isEmpty()", "in")]
        # Membership in a finite enumerated set is the disjunction of equality
        # with its members; `=` binds tighter than `or`, so no parens needed.
        choices = " or ".join(f"v = {m}" for m in members)
        return [(f"self.{pname}->forAll(v | {choices})", "in")]

    def _c_disjoint(self, ps, pname):
        q = self.g.value(ps, SH.disjoint)
        if not isinstance(q, URIRef):
            return []
        return [(f"self.{pname}->intersection(self.{self.canon.name(q)})->isEmpty()", "disjoint")]

    def _c_equals(self, ps, pname):
        q = self.g.value(ps, SH.equals)
        if not isinstance(q, URIRef):
            return []
        qn = self.canon.name(q)
        body = (f"self.{pname}->forAll(v | self.{qn}->includes(v)) and "
                f"self.{qn}->forAll(v | self.{pname}->includes(v))")
        return [(body, "equals")]

    def _c_compare(self, ps, pname, pred, op):
        q = self.g.value(ps, pred)
        if not isinstance(q, URIRef):
            return []
        qn = self.canon.name(q)
        return [(f"self.{pname}->forAll(v | self.{qn}->forAll(w | v {op} w))",
                 pred.split("#")[-1])]

    def _ocl_value(self, node):
        """OCL literal for ``sh:in`` / ``sh:hasValue`` members."""
        if isinstance(node, Literal):
            return self.m.ocl_literal(node)       # numeric -> bare, string -> '...'
        return self.canon.name(node)              # IRI -> UML name (individual/class)
