"""The transformation engine: applies rules D01-D41 and O01-O30 to build a
:class:`~owl2uml.model.UMLModel` from a parsed OWL-2 graph.

The :class:`Mapper` both orchestrates the phases (schema -> classes ->
properties -> restrictions -> individuals) and exposes the ``ensure_*`` /
``add_*`` helpers that :class:`~owl2uml.expressions.ExpressionResolver` calls
back into.
"""
from __future__ import annotations

from rdflib import URIRef, BNode, Literal
from rdflib.namespace import OWL, RDF, RDFS

from besser.BUML.metamodel.structural import data_types as _buml_data_types
from besser.BUML.notations.kg_to_buml._common import KGConversionWarning

from . import naming
from .axioms import AxiomIndex, primitive_for
from .canonicalize import Canonicalizer
from .expressions import ExpressionResolver
from .model import (
    UMLModel, Class, Attribute, DataType, Enumeration, Association,
    AssociationEnd, Generalization, InstanceSpecification, Slot, Link,
    OCLConstraint, Comment,
)

#: The nine BUML primitives. Deviation from the reference, which uses UML-ish
#: names; see :func:`..axioms.primitive_for`.
_PRIMITIVES = {p.name for p in _buml_data_types}


def _sorted_pairs(pairs):
    """Deterministic ordering for a ``(subject, object)`` rdflib result set."""
    return sorted(pairs, key=lambda t: (str(t[0]), str(t[1])))


class Mapper:
    """Applies rules D01-D41 and O01-O30, then optionally the SHACL S-rules.

    Deviations from the reference are limited to determinism (every rdflib
    result set is iterated in sorted order), the BUML datatype vocabulary, and
    routing diagnostics to a ``warnings`` sink instead of ``sys.stderr``. The
    reference's ``annotations`` / ``emit_external`` / ``enum_from_instances`` /
    ``neg_assertions`` options were accepted but never read, and ``emit_ocl``
    only gated PlantUML rendering; all five are dropped.
    """

    def __init__(self, graph, base, *, shapes=None, warnings=None):
        self.g = graph
        self.base = base
        self.ax = AxiomIndex(graph)
        self.canon = Canonicalizer(graph, base)
        self.model = UMLModel()
        self.shapes = shapes            # optional SHACL shapes graph (phase 2)
        self.warnings: list[KGConversionWarning] = warnings if warnings is not None else []
        self._name_to_iri: dict[str, URIRef] = {}
        self._gen_seen: set[tuple[str, str]] = set()
        self._handled_props: set = set()          # object props consumed by inverse pairs
        self._datarange_cache: dict = {}          # BNode -> DataType name (dedup shared anon data ranges)
        self._composite_datarange_cache: dict = {}  # (kind, members) -> DataType name (dedup by structure)
        self.resolver = ExpressionResolver(self)

        primary = (list(self.ax.sorted_named_classes) + list(self.ax.sorted_object_props)
                   + list(self.ax.sorted_data_props) + list(self.ax.sorted_individuals)
                   + list(self.ax.sorted_datatypes))
        self.canon.register_primary(primary)

    def warn(self, code: str, message: str) -> None:
        """Record a non-fatal diagnostic (the reference printed to stderr)."""
        self.warnings.append(KGConversionWarning(code=code, message=message))

    # =====================================================================
    #  orchestration
    # =====================================================================
    def run(self) -> UMLModel:
        self._map_ontology()          # D01-D04
        self._map_datatypes()         # D09, D37
        self._map_classes()           # D08
        self._map_class_axioms()      # D26, D27(via canon), O06, O16, O17, restriction †
        self._map_object_properties() # D10, D29, D30-D32, O22
        self._map_data_properties()   # D11, D34-D36
        self._map_sub_properties()    # O18, O27
        self._map_property_characteristics()  # O21, O23-O26
        self._map_individuals()       # D13, D39, D40, D41
        if self.shapes is not None:
            self._map_shacl()         # SHACL -> OCL (Table tab:shacl-ocl)
        self._post_checks()
        return self.model

    def _map_shacl(self) -> None:
        from .shacl import ShaclMapper   # local import to keep shacl.py optional
        ShaclMapper(self, self.shapes).run()

    # =====================================================================
    #  helpers used by the resolver + rules
    # =====================================================================
    def ensure_class(self, iri) -> str:
        name = self.canon.name(iri)
        if name in self.model.classes:
            return name
        rep = URIRef(self.canon.rep(iri))
        declared = rep in self.ax.named_classes()
        foreign = bool(self.base) and not str(rep).startswith(self.base)
        self.model.classes[name] = Class(
            name=name, uri=str(rep),
            is_stub=not declared,
            is_foreign=declared and foreign,
            is_deprecated=self.ax.is_deprecated(rep),
        )
        self._name_to_iri[name] = rep
        return name

    def make_aux_class(self, base_name: str, abstract: bool = False) -> str:
        name = self.canon.reserve(base_name)
        self.model.classes[name] = Class(name=name, is_auxiliary=True, is_abstract=abstract)
        return name

    def make_enum(self, literals: list[str], hint: str) -> str:
        name = self.canon.reserve(hint)
        seen, lits = set(), []
        for literal in literals:
            if literal not in seen:
                seen.add(literal)
                lits.append(literal)
        self.model.enumerations[name] = Enumeration(name=name, literals=lits)
        return name

    def iri_for_name(self, name: str):
        return self._name_to_iri.get(name)

    def add_generalization(self, sub: str, sup: str, **kw) -> None:
        if sub == sup or (sub, sup) in self._gen_seen:
            return
        self._gen_seen.add((sub, sup))
        self.model.generalizations.append(Generalization(subclass=sub, superclass=sup, **kw))

    def add_association(self, source_type: str, target_type: str, role: str,
                        src_mult=(0, "*"), tgt_mult=(0, "*"), name: str | None = None,
                        source_role: str | None = None, uri: str | None = None) -> None:
        self.model.associations.append(Association(
            name=name or role,
            uri=uri,
            source=AssociationEnd(type=source_type, role=source_role,
                                  lower=src_mult[0], upper=src_mult[1],
                                  navigable=source_role is not None),
            target=AssociationEnd(type=target_type, role=role,
                                  lower=tgt_mult[0], upper=tgt_mult[1], navigable=True),
        ))

    def add_ocl(self, context: str, body: str, origin: str = "",
                kind: str = "inv", name: str | None = None) -> None:
        if context not in self.model.classes:
            # context must exist to carry the note (e.g. Thing)
            self.model.classes[context] = Class(name=context, is_stub=True)
        self.model.classes[context].invariants.append(
            OCLConstraint(context=context, body=body, kind=kind, name=name, origin_rule=origin)
        )

    def set_attribute_type(self, ctx: str, pname: str, dtype: str) -> bool:
        """Retarget an already-declared Attribute's type.

        Used by the SHACL phase when a ``sh:or``/``sh:and``/``sh:not`` constraint
        over datatypes reveals the same union/intersection/complement type the OWL
        phase already materialised for this property (a no-op re-assignment) — or,
        if the OWL side never declared a precise ``rdfs:range`` for it, the more
        precise type the shape constraint implies. Returns ``False`` (no-op) if no
        matching attribute is found on ``ctx``.
        """
        cls = self.model.classes.get(ctx)
        if cls is None:
            return False
        for attr in cls.attributes:
            if attr.name == pname:
                attr.type = dtype
                return True
        return False

    def ensure_instance(self, iri) -> str:
        name = self.canon.name(iri)
        if name in self.model.instances:
            return name
        inst = InstanceSpecification(name=name, uri=str(iri))
        self.model.instances[name] = inst
        cls = self.ax.classifier_of(iri)
        inst.classifier = self.ensure_class(cls) if cls is not None else None
        for p, lit in self.ax.data_assertions(iri):
            inst.slots.append(Slot(attribute=self.canon.name(p), values=[self.literal_value(lit)]))
        return name

    def classifier_name(self, iri) -> str:
        cls = self.ax.classifier_of(iri)
        return self.ensure_class(cls) if cls is not None else self.ensure_class(OWL.Thing)

    def literal_value(self, node) -> str:
        if isinstance(node, Literal):
            v = node.value
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            return '"' + str(node).replace('"', '\\"').replace("\n", " ") + '"'
        return '"' + str(node) + '"'

    def register_primitive(self, name: str) -> str:
        if name not in self.model.datatypes:
            self.model.datatypes[name] = DataType(name=name, is_primitive=True)
        return name

    def resolve_data_range(self, node) -> str:
        if node is None:
            return self.register_primitive(primitive_for(None))
        if isinstance(node, URIRef):
            if node in self.ax.datatypes():          # custom rdfs:Datatype (D09/D37)
                dt = self.canon.name(node)
                if dt not in self.model.datatypes:
                    self.model.datatypes[dt] = DataType(name=dt, uri=str(node))
                self._name_to_iri.setdefault(dt, node)
                return dt
            return self.register_primitive(primitive_for(node))
        if isinstance(node, BNode):                  # O01-O05 data-range expression
            if node in self._datarange_cache:        # same anon node -> same DataType (no dup classes)
                return self._datarange_cache[node]
            name = self._materialize_data_range(node)
            self._datarange_cache[node] = name
            return name
        return self.register_primitive("String")

    def _materialize_data_range(self, node) -> str:
        base = self.g.value(node, OWL.onDatatype)
        if base is not None:                          # DatatypeRestriction (O05)
            base_name = self.resolve_data_range(base)
            facets: list[str] = []
            restr = self.g.value(node, OWL.withRestrictions)
            if restr is not None:
                from rdflib.collection import Collection as RDFList
                for f in RDFList(self.g, restr):
                    for fp, fo in self.g.predicate_objects(f):
                        facets.append(f"{naming.local_name(fp)} {self.literal_value(fo)}")
            key = ("restriction", base_name, tuple(facets))
            cached = self._composite_datarange_cache.get(key)
            if cached:
                return cached
            name = self.canon.reserve(f"_{base_name}_restricted")
            self.model.datatypes[name] = DataType(name=name, base=base_name, facets=facets)
            self._composite_datarange_cache[key] = name
            return name
        from rdflib.collection import Collection as RDFList
        # O01-O03: resolve every operand so structurally-identical data ranges
        # (a repeated "unionOf(xsd:dateTime, xsd:date)" range, re-wrapped in a
        # *fresh* blank node per property, is the common case) collapse onto
        # the SAME materialised DataType instead of minting one per occurrence
        # (see ``composite_datarange``).
        lst = self.g.value(node, OWL.unionOf)
        if lst is not None:                            # O02 DataUnionOf
            members = [self.resolve_data_range(m) for m in RDFList(self.g, lst)]
            return self.composite_datarange(members, "union")
        lst = self.g.value(node, OWL.intersectionOf)
        if lst is not None:                            # O01 DataIntersectionOf
            members = [self.resolve_data_range(m) for m in RDFList(self.g, lst)]
            return self.composite_datarange(members, "intersection")
        comp = self.g.value(node, OWL.datatypeComplementOf)
        if comp is not None:                           # O03 DataComplementOf
            return self.composite_datarange([self.resolve_data_range(comp)], "complement")
        # owl:oneOf of typed literals (O04) or anything unrecognised -> generic fallback
        return self.composite_datarange([], "opaque")

    def composite_datarange(self, members: list[str], kind: str) -> str:
        """Materialise (or reuse) the DataType for a union/intersection/complement
        data-range expression (O01-O03).

        Deduplicated by its *structure* — ``kind`` plus the *set* of member
        names, order-independent — rather than by blank-node identity or list
        order: an inline ``[ owl:unionOf (...) ]`` expression parses to a fresh
        :class:`BNode` at every occurrence even when the operands are identical
        (so identity-only caching would still fan out into one duplicate
        DataType per occurrence), and the SAME set of XSD types can be listed in
        a different order by a SHACL ``sh:or`` than by the OWL ``owl:unionOf`` it
        mirrors — sorting the key ensures both collapse onto one shared type.

        Rather than encoding the constituent types as inert facet strings (which
        would then force every class that uses this type to repeat an identical
        "is one of these types" OCL check), the DataType carries a single
        ``value : Any`` attribute plus one OCL invariant — attached to the
        DataType itself — verifying that attribute's type membership. Classes
        using this DataType need no invariant of their own; the guarantee lives
        where the constraint actually belongs.
        """
        # BUML's datatype vocabulary is coarser than the paper's (anyURI,
        # gYear and gYearMonth all collapse onto ``str``), so an OWL union of
        # four XSD types can arrive here with repeated members. Deduplicate
        # before building the name and the OCL body, otherwise the invariant
        # reads "... or v.oclIsTypeOf(str) or v.oclIsTypeOf(str)".
        members = list(dict.fromkeys(members))
        key = (kind, tuple(sorted(set(members))))
        cached = self._composite_datarange_cache.get(key)
        if cached:
            return cached
        if kind == "union":
            hint = naming.union_name(members)
        elif kind == "intersection":
            hint = naming.intersection_name(members)
        elif kind == "complement" and members:
            hint = naming.not_name(members[0])
        else:
            hint = "_DataRange"
        name = self.canon.reserve(hint)
        dt = DataType(name=name)
        if kind in ("union", "intersection", "complement") and members:
            # "any" (lowercase) is the BUML primitive; see ``axioms.primitive_for``.
            dt.attributes.append(Attribute(name="value", type="any", lower=0, upper="*"))
            if kind == "union":
                expr = " or ".join(f"v.oclIsTypeOf({m})" for m in members)
                origin = "O02"
            elif kind == "intersection":
                expr = " and ".join(f"v.oclIsTypeOf({m})" for m in members)
                origin = "O01"
            else:                                      # complement (single operand)
                expr = f"not v.oclIsTypeOf({members[0]})"
                origin = "O03"
            dt.invariants.append(OCLConstraint(
                context=name, body=f"self.value->forAll(v | {expr})",
                kind="inv", name=f"{kind}_invariant", origin_rule=origin,
            ))
        self.model.datatypes[name] = dt
        self._composite_datarange_cache[key] = name
        return name

    def satisfies(self, dr: str, var: str) -> str:
        return f"{var}.oclIsKindOf({dr})"

    # =====================================================================
    #  phases
    # =====================================================================
    def _map_ontology(self) -> None:
        onto = self.ax.ontology()
        if onto is None:
            self.model.name = "ontology"
            return
        self.model.uri = str(onto)
        self.model.name = naming.sanitize(naming.local_name(onto))
        ver = self.g.value(onto, OWL.versionIRI) or self.g.value(onto, OWL.versionInfo)
        if ver is not None:
            self.model.version = str(ver)
        for imp in sorted(self.g.objects(onto, OWL.imports), key=str):
            self.model.imports.append(str(imp))
        parts = []
        for pred in (URIRef("http://purl.org/dc/terms/title"),
                     URIRef("http://purl.org/dc/elements/1.1/title"),
                     RDFS.comment,
                     URIRef("http://purl.org/dc/terms/description")):
            for o in sorted(self.g.objects(onto, pred), key=str):
                parts.append(str(o).strip())
        if parts:
            self.model.comments.append(Comment(text=" — ".join(parts)))

    def _map_datatypes(self) -> None:
        for dt in self.ax.sorted_datatypes:
            if self.canon.is_alias(dt):
                continue
            self.resolve_data_range(dt)

    def _map_classes(self) -> None:
        for c in self.ax.sorted_named_classes:
            self.ensure_class(c)

    def _map_class_axioms(self) -> None:
        # D26 subClassOf (named or expression); restriction/union superclasses via resolver
        for sub, sup in self.ax.subclass_axioms():
            sub_name = self.ensure_class(sub)
            if isinstance(sup, URIRef):
                self.add_generalization(sub_name, self.ensure_class(sup))
            else:
                self.resolver.resolve_class(sup, subsuming=sub_name)
        # equivalentClass to an anonymous expression -> treat like subclass wiring
        for s, o in self.ax.equivalent_classes():
            if isinstance(s, URIRef) and isinstance(o, BNode):
                self.resolver.resolve_class(o, subsuming=self.ensure_class(s))
        # O16 DisjointClasses (pairwise OCL)
        for a, b in _sorted_pairs(self.g.subject_objects(OWL.disjointWith)):
            if isinstance(a, URIRef) and isinstance(b, URIRef):
                self.add_ocl(self.ensure_class(a),
                             f"not self.oclIsKindOf({self.ensure_class(b)})", origin="O16")
        # O17 DisjointUnion
        for c, lst in _sorted_pairs(self.g.subject_objects(OWL.disjointUnionOf)):
            from rdflib.collection import Collection as RDFList
            members = [self.resolver.resolve_class(m) for m in RDFList(self.g, lst)]
            cname = self.ensure_class(c)
            for m in members:
                self.add_generalization(m, cname, set_name="disjointUnion",
                                        is_disjoint=True, is_complete=True)
            disj = " or ".join(f"self.oclIsKindOf({m})" for m in members)
            self.add_ocl(cname, disj, origin="O17", name="complete")

    def _combine(self, nodes) -> str:
        """Resolve one-or-more domain/range nodes to a single class name."""
        names = sorted({self.resolver.resolve_class(n) for n in nodes})
        if not names:
            return self.ensure_class(OWL.Thing)
        if len(names) == 1:
            return names[0]
        aux = self.make_aux_class(naming.intersection_name(names))
        for nm in names:
            self.add_generalization(aux, nm)
        return aux

    def _participating_classes(self, nodes) -> list[str]:
        """Classes an object-property association (or its O22 invariant)
        attaches to.

        Per the spec, an ObjectPropertyDomain / ObjectPropertyRange that is a
        union of classes is linked directly to each participating class rather
        than through an auxiliary union class; a member that is a (transitive)
        subclass of another member is dropped, since the superclass link
        already covers it. Non-union expressions resolve to a single (possibly
        materialised) class, and an absent domain/range falls back to Thing.
        """
        if not nodes:
            return [self.ensure_class(OWL.Thing)]
        names: list[str] = []
        for n in nodes:
            members = self.resolver.union_members(n)
            names.extend(members if members is not None else [self.resolver.resolve_class(n)])
        return self._drop_subclasses(list(dict.fromkeys(names)))

    def _drop_subclasses(self, names: list[str]) -> list[str]:
        nameset = set(names)
        kept: list[str] = []
        for n in names:
            iri = self.iri_for_name(n)
            ancestors = ({self.canon.name(s)
                          for s in self.g.transitive_objects(iri, RDFS.subClassOf)
                          if isinstance(s, URIRef) and s != iri}
                         if iri is not None else set())
            if ancestors & nameset:
                continue                 # a superclass of n is also present -> redundant link
            kept.append(n)
        return kept

    def _warn_missing_domain_range(self, prop, domains, ranges) -> None:
        """Surface the two fallbacks that silently widen a property.

        Neither is fatal — a domain-less property attaches to ``Thing`` and a
        range-less data property becomes a string — but both mean the generated
        model is less precise than the user probably intended, and the KG
        refinement flow offers a fix for each.
        """
        name = naming.local_name(prop)
        if not domains:
            self.warn(
                "PROPERTY_NO_DOMAIN",
                f"Property '{name}' has no rdfs:domain; attached to Thing.",
            )
        if not ranges:
            self.warn(
                "PROPERTY_NO_RANGE",
                f"Property '{name}' has no rdfs:range; defaulted to the widest type.",
            )

    def _group_domains(self, prop):
        nodes = []
        for member in self.canon.group(prop):
            nodes += self.ax.domains(URIRef(member))
        return nodes

    def _group_ranges(self, prop):
        nodes = []
        for member in self.canon.group(prop):
            nodes += self.ax.ranges(URIRef(member))
        return nodes

    def _map_object_properties(self) -> None:
        # D29 inverse pairs first (one bidirectional association per member pair)
        for a, b in self.ax.inverse_pairs():
            if not (isinstance(a, URIRef) and isinstance(b, URIRef)):
                continue
            if a in self._handled_props or b in self._handled_props:
                continue
            self._handled_props.update({a, b})
            srcs = self._participating_classes(self._group_domains(a) or self._group_ranges(b))
            tgts = self._participating_classes(self._group_ranges(a) or self._group_domains(b))
            for src in srcs:
                for tgt in tgts:
                    self.add_association(src, tgt, role=self.canon.name(a),
                                         source_role=self.canon.name(b), uri=str(a))
        # remaining object properties
        for p in self.ax.sorted_object_props:
            if self.canon.is_alias(p) or p in self._handled_props:
                continue
            self._warn_missing_domain_range(p, self._group_domains(p), self._group_ranges(p))
            srcs = self._participating_classes(self._group_domains(p))
            tgts = self._participating_classes(self._group_ranges(p))
            tgt_mult = (0, 1) if self.ax.is_functional(p) else (0, "*")
            src_mult = (0, 1) if self.ax.is_inverse_functional(p) else (0, "*")
            pname = self.canon.name(p)
            for src in srcs:
                for tgt in tgts:
                    self.add_association(src, tgt, role=pname,
                                         src_mult=src_mult, tgt_mult=tgt_mult, uri=str(p))
            if self.ax.is_inverse_functional(p):     # O22
                for src in srcs:
                    for tgt in tgts:
                        self.add_ocl(tgt, f"{tgt}.allInstances()->forAll(d | {src}.allInstances()"
                                          f"->select(c | c.{pname}->includes(d))->size() <= 1)",
                                     origin="O22")

    def _map_data_properties(self) -> None:
        for p in self.ax.sorted_data_props:
            if self.canon.is_alias(p):
                continue
            pname = self.canon.name(p)
            ranges = self._group_ranges(p)
            self._warn_missing_domain_range(p, self._group_domains(p), ranges)
            dtype = (self.resolve_data_range(ranges[0]) if ranges
                     else self.register_primitive(primitive_for(None)))
            upper = 1 if self.ax.is_functional(p) else "*"
            domains = self._group_domains(p)
            owner = self._combine(domains) if domains else self.ensure_class(OWL.Thing)
            cls = self.model.classes[owner]
            if any(a.name == pname for a in cls.attributes):
                continue
            cls.attributes.append(Attribute(name=pname, type=dtype, lower=0, upper=upper,
                                            uri=str(p)))

    def _has_property(self, cls_name: str, pname: str) -> bool:
        """Whether `pname` is available — as an attribute or a navigable
        association role — on `cls_name` or (transitively) on any of its
        superclasses.

        Used to guard OCL constraints (O18/O27, SHACL) that reference a
        property: some ontologies declare a rdfs:subPropertyOf superproperty
        (or a SHACL shape's sh:path) that was never itself declared with a
        domain/range/characteristics, so it never materialises as an
        attribute or association anywhere. Emitting ``self.<pname>`` in that
        case would reference a feature the context class doesn't have, so the
        constraint is skipped instead.

        Every OWL class is implicitly ``rdfs:subClassOf owl:Thing`` even when
        no explicit generalization edge encodes it, so a domain-less
        property/data-property (which the mapper falls back to attaching to
        ``Thing``, see ``_map_data_properties``/``_map_object_properties``) is
        reachable from *any* class — ``Thing`` is always checked as an
        implicit ancestor in addition to the walked generalization chain.
        """
        thing_name = self.canon.name(OWL.Thing)
        seen: set[str] = set()
        stack = [cls_name, thing_name]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            cls = self.model.classes.get(name)
            if cls is not None and any(a.name == pname for a in cls.attributes):
                return True
            for assoc in self.model.associations:
                if assoc.source.type == name and assoc.target.role == pname:
                    return True
                if (assoc.target.type == name and assoc.source.navigable
                        and assoc.source.role == pname):
                    return True
            for gen in self.model.generalizations:
                if gen.subclass == name:
                    stack.append(gen.superclass)
        return False

    def _map_sub_properties(self) -> None:
        for sub, sup in self.ax.sub_property_axioms():
            subn, supn = self.canon.name(sub), self.canon.name(sup)
            if subn == supn:
                continue
            if self.ax.is_object_property(sub):                     # O18
                ctx = self._combine(self._group_domains(sub)) if self._group_domains(sub) \
                    else self.ensure_class(OWL.Thing)
                if not self._has_property(ctx, supn):
                    continue
                self.add_ocl(ctx, f"self.{subn}->forAll(o | self.{supn}->includes(o))", origin="O18")
            elif self.ax.is_data_property(sub):                     # O27
                ctx = self._combine(self._group_domains(sub)) if self._group_domains(sub) \
                    else self.ensure_class(OWL.Thing)
                if not self._has_property(ctx, supn):
                    continue
                self.add_ocl(ctx, f"self.{subn}->asSet()->forAll(v | self.{supn}->asSet()->includes(v))",
                             origin="O27")

    def _map_property_characteristics(self) -> None:
        rules = [
            (OWL.ReflexiveProperty,  "self.{p}->includes(self)", "O23"),
            (OWL.IrreflexiveProperty, "self.{p}->excludes(self)", "O24"),
            (OWL.AsymmetricProperty, "self.{p}->forAll(x | not x.{p}->includes(self))", "O25"),
            (OWL.TransitiveProperty, "self.{p}->forAll(x | x.{p}->forAll(y | self.{p}->includes(y)))", "O26"),
        ]
        for rdftype, tmpl, origin in rules:
            for p in sorted(self.g.subjects(RDF.type, rdftype), key=str):
                if not isinstance(p, URIRef):
                    continue
                pname = self.canon.name(p)
                ctx = self._combine(self._group_domains(p)) if self._group_domains(p) \
                    else self.ensure_class(OWL.Thing)
                self.add_ocl(ctx, tmpl.format(p=pname), origin=origin)
        # O21 DisjointObjectProperties (pairwise)
        for a, b in _sorted_pairs(self.g.subject_objects(OWL.propertyDisjointWith)):
            if isinstance(a, URIRef) and isinstance(b, URIRef) and self.ax.is_object_property(a):
                an, bn = self.canon.name(a), self.canon.name(b)
                ctx = self._combine(self._group_domains(a)) if self._group_domains(a) \
                    else self.ensure_class(OWL.Thing)
                self.add_ocl(ctx, f"self.{an}->forAll(o | self.{bn}->excludes(o))", origin="O21")

    def _map_individuals(self) -> None:
        for i in self.ax.sorted_individuals:
            name = self.ensure_instance(i)
            for p, o in self.ax.object_assertions(i):       # D40
                tgt = self.ensure_instance(o)
                self.model.links.append(Link(source=name, target=tgt, role=self.canon.name(p)))

    def _post_checks(self) -> None:
        """Create stubs for any type referenced but not yet declared."""
        known = self.model.all_type_names()
        referenced: set[str] = set()
        for a in self.model.associations:
            referenced.update({a.source.type, a.target.type})
        for cls in self.model.classes.values():
            for attr in cls.attributes:
                referenced.add(attr.type)
        for g in self.model.generalizations:
            referenced.update({g.subclass, g.superclass})
        for inst in self.model.instances.values():
            if inst.classifier:
                referenced.add(inst.classifier)
        for missing in sorted(referenced - known - _PRIMITIVES):
            if missing and missing not in self.model.classes:
                self.model.classes[missing] = Class(name=missing, is_stub=True)
