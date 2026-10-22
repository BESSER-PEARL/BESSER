"""The transformation engine: applies rules D01-D41 and O01-O30 to build a
``ModelPlan`` from a parsed OWL-2 graph.

The :class:`Mapper` both orchestrates the phases (schema -> classes ->
properties -> restrictions -> individuals) and exposes the ``ensure_*`` /
``add_*`` helpers that :class:`~owl2uml.expressions.ExpressionResolver` calls
back into.
"""
from __future__ import annotations

from rdflib import URIRef, BNode, Literal
from rdflib.collection import Collection as RDFList
from rdflib.namespace import OWL, RDF, RDFS, XSD

from besser.BUML.metamodel.structural import data_types as _buml_data_types
from besser.BUML.notations.kg_to_buml._common import KGConversionWarning

from . import naming
from .axioms import AxiomIndex, primitive_for
from .canonicalize import Canonicalizer
from .expressions import ExpressionResolver
from ._plan import (
    ModelPlan, PlannedClass, PlannedAttribute, PlannedDataType, PlannedEnumeration, PlannedAssociation,
    PlannedInvariant,
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
        self.model = ModelPlan()
        self.shapes = shapes            # optional SHACL shapes graph (phase 2)
        self.warnings: list[KGConversionWarning] = warnings if warnings is not None else []
        self._name_to_iri: dict[str, URIRef] = {}
        self._gen_seen: set[tuple[str, str]] = set()
        self._instances_seen: set[str] = set()    # individuals already named (see ensure_instance)
        self._handled_props: set = set()          # object props consumed by inverse pairs
        self._datarange_cache: dict = {}          # BNode -> DataType name (dedup shared anon data ranges)
        self._composite_datarange_cache: dict = {}  # (kind, members) -> DataType name (dedup by structure)
        self._enum_defs: dict[URIRef, BNode] = {}  # D20: named class -> the oneOf node it is an enumeration of
        self._datatypes_in_progress: set[str] = set()  # D37: datatype names whose definition is being followed
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
    def run(self) -> ModelPlan:
        """Run every mapping phase over the graph and return the finished plan.

        Returns:
            ModelPlan: The classes, associations, generalizations and invariants the rules produced.
        """
        self._map_enum_classes()      # D20
        self._map_datatypes()         # D09, D17, D37, O04, O05
        self._map_classes()           # D08
        self._map_class_axioms()      # D26, D27(via canon), O06, O16, O17, restriction †
        self._map_object_properties() # D10, D29, D30-D32, O22
        self._map_data_properties()   # D11, D34-D36
        self._map_sub_properties()    # O18, O27
        self._map_property_chains()   # O19, O20
        self._map_property_characteristics()  # O21, O23-O26, O28
        self._map_keys()              # D38
        self._map_individuals()       # D13, D39, D40, D41
        if self.shapes is not None:
            self._map_shacl()         # SHACL -> OCL (Table tab:shacl-ocl)
        self._complete_thing_hierarchy()
        self._post_checks()
        return self.model

    def _complete_thing_hierarchy(self) -> None:
        """Parent the classes the KG could not name under ``Thing``.

        ``attach_to_thing`` — the Refine-KG resolution offered for
        ``PROPERTY_NO_DOMAIN`` — makes every top-level *KG* class a subclass of
        Thing, which is what lets an invariant navigate a domain-less property:
        OCL resolves ``self.<p>`` by walking the context's ancestors, so Thing
        has to really be one. The classes materialised during this mapping —
        union and intersection aux classes above all — do not exist at KG time,
        so they would be left as roots and any invariant whose context is one of
        them would still fail to resolve.

        Runs only when the KG already puts a class under Thing, i.e. the user
        accepted the recommendation. Declining leaves the model as it was.

        One exception: a class that declares a feature name Thing also declares
        keeps its own. Restriction aux classes exist precisely to narrow such a
        property (``_all_hasPart_Issue --hasPart--> Issue``, ``_max_1_isPartOf_Thing``),
        and BUML keeps only one end per name across a hierarchy — so parenting
        them would drop the narrower end and, with it, the cardinality that only
        the association carries.
        """
        thing = self.canon.name(OWL.Thing)
        if thing not in self.model.classes:
            return
        if not any(sup == thing for _, sup in self.model.generalizations):
            return

        def features(name: str) -> set[str]:
            cls = self.model.classes.get(name)
            names = {a.name for a in cls.attributes} if cls is not None else set()
            for assoc in self.model.associations:
                if assoc.source == name and assoc.role:
                    names.add(assoc.role)
                if assoc.target == name and assoc.navigable_source and assoc.source_role:
                    names.add(assoc.source_role)
            return names

        inherited = features(thing)
        has_parent = {sub for sub, _ in self.model.generalizations}
        for name in sorted(self.model.classes):
            if name == thing or name in has_parent:
                continue
            if features(name) & inherited:
                continue
            self.add_generalization(name, thing)

    def _map_shacl(self) -> None:
        from .shacl import ShaclMapper   # local import to keep shacl.py optional
        ShaclMapper(self, self.shapes).run()

    # =====================================================================
    #  helpers used by the resolver + rules
    # =====================================================================
    def ensure_class(self, iri) -> str:
        """Return the plan name of the class ``iri``, declaring it on first use.

        An IRI the ontology never declares becomes a stub (carried as ``external``).
        """
        name = self.canon.name(iri)
        if name in self.model.classes or name in self.model.enumerations:
            return name
        rep = URIRef(self.canon.rep(iri))
        declared = rep in self.ax.named_classes()
        foreign = bool(self.base) and not str(rep).startswith(self.base)
        self.model.classes[name] = PlannedClass(
            name=name, uri=str(rep),
            is_stub=not declared,
            is_foreign=declared and foreign,
            is_deprecated=self.ax.is_deprecated(rep),
        )
        self._name_to_iri[name] = rep
        return name

    def make_aux_class(self, base_name: str, abstract: bool = False) -> str:
        """Declare an auxiliary class (union, restriction, ...) under a fresh reserved name and return it."""
        name = self.canon.reserve(base_name)
        self.model.classes[name] = PlannedClass(name=name, is_auxiliary=True, is_abstract=abstract)
        return name

    def make_enum(self, literals: list[str], hint: str, exact_name: str | None = None) -> str:
        """Declare an enumeration of ``literals`` (duplicates dropped, order kept) and return its name.

        The name is reserved from ``hint``, unless ``exact_name`` is given: the
        canonicaliser already reserved it for a named class or datatype that
        becomes the enumeration itself (D20, D37).
        """
        name = exact_name or self.canon.reserve(hint)
        seen, lits = set(), []
        for literal in literals:
            if literal not in seen:
                seen.add(literal)
                lits.append(literal)
        self.model.enumerations[name] = PlannedEnumeration(name=name, literals=lits)
        return name

    def iri_for_name(self, name: str) -> URIRef | None:
        """Return the IRI a plan name was minted for, or ``None`` for one with no IRI (an auxiliary class)."""
        return self._name_to_iri.get(name)

    def add_generalization(self, sub: str, sup: str) -> None:
        """Record ``sub`` as a subclass of ``sup``; a self-pair or a repeat is ignored."""
        if sub == sup or (sub, sup) in self._gen_seen:
            return
        self._gen_seen.add((sub, sup))
        self.model.generalizations.append((sub, sup))

    def add_association(self, source_type: str, target_type: str, role: str,
                        src_mult=(0, "*"), tgt_mult=(0, "*"), name: str | None = None,
                        source_role: str | None = None, uri: str | None = None) -> None:
        """Record a binary association ``source_type --role--> target_type``.

        Args:
            role: The target end's name.
            src_mult: ``(lower, upper)`` of the source end.
            tgt_mult: ``(lower, upper)`` of the target end.
            name: Association name; defaults to ``role``.
            source_role: The reverse end's name. Giving one makes that end navigable.
            uri: IRI of the property the association came from.
        """
        self.model.associations.append(PlannedAssociation(
            name=name or role,
            source=source_type,
            target=target_type,
            role=role,
            source_role=source_role,
            src_mult=tuple(src_mult),
            tgt_mult=tuple(tgt_mult),
            navigable_source=source_role is not None,
            uri=uri,
        ))

    def add_feature(self, source: str, target: str, role: str, *, src_mult=(0, "*"), tgt_mult=(0, "*"),
                    uri: str | None = None, **association) -> None:
        """Link ``source`` to ``target`` through ``role``: an association, or an attribute when ``target`` is an enumeration (D20).

        An enumeration is a type, not a class: it has no end to navigate back
        from, so a property ranging over one is an attribute of its source.
        """
        if target in self.model.enumerations:
            self.add_attribute(source, role, target, tgt_mult, uri=uri)
        else:
            self.add_association(source, target, role, src_mult, tgt_mult, uri=uri, **association)

    def add_attribute(self, ctx: str, pname: str, dtype: str,
                      mult=(0, "*"), uri: str | None = None) -> None:
        """Declare an attribute on ``ctx``, unless it already has one by that name.

        The data-property counterpart of :meth:`add_association`. A restriction
        over a data property materialises an auxiliary class the same way one
        over an object property does, and the class has to own the feature it
        constrains — otherwise the invariant it carries names something that
        lives on its subclasses, where no OCL evaluator will look for it.
        """
        cls = self.model.classes.get(ctx)
        if cls is None or any(a.name == pname for a in cls.attributes):
            return
        cls.attributes.append(
            PlannedAttribute(name=pname, type=dtype, lower=mult[0], upper=mult[1], uri=uri)
        )

    def add_ocl(self, context: str, body: str, origin: str = "",
                name: str | None = None) -> None:
        """Attach an OCL invariant ``body`` to ``context``, declaring the context as a stub if it does not exist.

        Args:
            origin: Id of the mapping rule that produced it.
            name: Optional constraint name.
        """
        if context not in self.model.classes:
            # context must exist to carry the note (e.g. Thing)
            self.model.classes[context] = PlannedClass(name=context, is_stub=True)
        self.model.classes[context].invariants.append(
            PlannedInvariant(context=context, body=body, name=name, origin_rule=origin)
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
        """Name an individual and make sure its classifier exists.

        No instance is recorded: nothing downstream consumes one. What this still
        does matters, though, and is easy to mistake for dead code:

        * ``canon.name`` reserves the individual's name (and its data properties'),
          so the order of first requests decides which IRI keeps an un-suffixed
          name when two collide;
        * ``ensure_class`` creates a stub for a classifier the ontology never
          declared, which the output carries as an ``external`` class.
        """
        name = self.canon.name(iri)
        if name in self._instances_seen:
            return name
        self._instances_seen.add(name)
        cls = self.ax.classifier_of(iri)
        if cls is not None:
            self.ensure_class(cls)
        for p, _literal in self.ax.data_assertions(iri):
            self.canon.name(p)
        return name

    def classifier_name(self, iri) -> str:
        """Return the plan name of the class an individual is asserted to be, falling back to ``owl:Thing``."""
        cls = self.ax.classifier_of(iri)
        return self.ensure_class(cls) if cls is not None else self.ensure_class(OWL.Thing)

    def literal_value(self, node) -> str:
        """Render ``node`` for a *non-OCL* position: instance slots, datatype
        facets, enumeration literal names.

        Use :meth:`ocl_literal` for anything that ends up inside an invariant —
        B-OCL only accepts single-quoted strings.
        """
        if isinstance(node, Literal):
            v = node.value
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            return '"' + str(node).replace('"', '\\"').replace("\n", " ") + '"'
        return '"' + str(node) + '"'

    def ocl_literal(self, node) -> str:
        """Render ``node`` as a B-OCL literal.

        Numbers and booleans are bare, exactly as in :meth:`literal_value`;
        strings are single-quoted because ``STRING_LITERAL`` in ``BOCL.g4``
        accepts no other quote character. Backslash and quote are escaped —
        unlike :func:`~besser.BUML.notations.kg_to_buml.owl2uml.naming.regex_literal`,
        which deliberately keeps backslashes raw so regexes survive intact.
        """
        if isinstance(node, Literal):
            v = node.value
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
        text = str(node).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
        return "'" + text + "'"

    def register_primitive(self, name: str) -> str:
        """Declare the primitive datatype ``name`` once and return it."""
        if name not in self.model.datatypes:
            self.model.datatypes[name] = PlannedDataType(name=name, is_primitive=True)
        return name

    def resolve_data_range(self, node) -> str:
        """Return the plan name of the data range ``node``.

        Handles a custom ``rdfs:Datatype`` (with its definition, D37), a standard XSD IRI (mapped to a BUML primitive), and an anonymous data-range expression (materialised, once per node). ``None`` and anything else fall back to a primitive.
        """
        if node is None:
            return self.register_primitive(primitive_for(None))
        if isinstance(node, URIRef):
            if node in self.ax.datatypes():          # custom rdfs:Datatype (D09/D37)
                dt = self.canon.name(node)
                if dt in self._datatypes_in_progress:
                    # Reached again while its own definition is being followed
                    # (``A ≡ unionOf(B, …)``, ``B ≡ unionOf(A, …)``). The name is
                    # registered and the outer call finishes the type, so this
                    # reference just uses the name.
                    self.warn("DATATYPE_CYCLIC",
                              f"Datatype '{dt}' is defined in terms of itself; OWL 2 requires "
                              f"datatype definitions to be acyclic. The definition is followed "
                              f"once and the inner reference kept as a reference to '{dt}'.")
                elif dt not in self.model.datatypes and dt not in self.model.enumerations:
                    self._datatypes_in_progress.add(dt)
                    self._define_datatype(node, dt)
                    self._datatypes_in_progress.discard(dt)
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

    def _datatype_definition(self, dt):
        """The data range a custom datatype is defined as (D37), or ``None`` for a bare declaration.

        Read off every IRI the canonicaliser folded into ``dt``'s group (D27), so
        a definition stated on an alias is not lost. An ``owl:equivalentClass``
        to another custom datatype of that same group (``A ≡ A``, ``A ≡ B ≡ A``,
        a plain ``A ≡ B``) is the merge itself, not a definition: following it
        would make the datatype a ``value`` wrapper of itself, or recurse
        forever. (An XSD datatype in the group, ``A ≡ xsd:string``, is a real
        definition.)
        """
        group = set(self.canon.group(dt))
        candidates = {
            definition
            for member in group
            for definition in self.g.objects(URIRef(member), OWL.equivalentClass)
            if not (str(definition) in group and definition in self.ax.datatypes())
        }
        for definition in sorted(candidates, key=str):
            if isinstance(definition, BNode) or definition in self.ax.datatypes() \
                    or str(definition).startswith(str(XSD)):
                return definition
        return None

    def _define_datatype(self, node, name: str) -> None:
        """Declare the custom datatype ``node`` under ``name`` (D09), materialising its definition (D37).

        The datatype keeps its own name, so every attribute typed by it reads
        ``Age18`` and not ``_int_restricted``. A definition that is just another
        datatype becomes a ``value`` slot of that type, with no invariant.
        """
        definition = self._datatype_definition(node)
        if isinstance(definition, BNode):
            self._materialize_data_range(definition, name=name)
        elif definition is not None:
            target = self.resolve_data_range(definition)
            dt = PlannedDataType(name=name)
            dt.attributes.append(PlannedAttribute(name="value", type=target, lower=0, upper="*"))
            self.model.datatypes[name] = dt
        else:
            self.model.datatypes[name] = PlannedDataType(name=name)
        defined = self.model.datatypes.get(name) or self.model.enumerations.get(name)
        if defined is not None:
            defined.uri = str(node)

    def _materialize_data_range(self, node, name: str | None = None) -> str:
        """Materialise the anonymous data range ``node`` (O01-O05, D17), under ``name`` when one is given."""
        base = self.g.value(node, OWL.onDatatype)
        if base is not None:                          # DatatypeRestriction (O05)
            return self._restricted_datatype(node, base, name)
        # O01-O03: resolve every operand so structurally-identical data ranges
        # (a repeated "unionOf(xsd:dateTime, xsd:date)" range, re-wrapped in a
        # *fresh* blank node per property, is the common case) collapse onto
        # the SAME materialised DataType instead of minting one per occurrence
        # (see ``composite_datarange``).
        lst = self.g.value(node, OWL.unionOf)
        if lst is not None:                            # O02 DataUnionOf
            members = [self.resolve_data_range(m) for m in RDFList(self.g, lst)]
            return self.composite_datarange(members, "union", name=name)
        lst = self.g.value(node, OWL.intersectionOf)
        if lst is not None:                            # O01 DataIntersectionOf
            members = [self.resolve_data_range(m) for m in RDFList(self.g, lst)]
            return self.composite_datarange(members, "intersection", name=name)
        comp = self.g.value(node, OWL.datatypeComplementOf)
        if comp is not None:                           # O03 DataComplementOf
            return self.composite_datarange([self.resolve_data_range(comp)], "complement", name=name)
        lst = self.g.value(node, OWL.oneOf)
        if lst is not None:                            # D17 / O04 DataOneOf
            return self._data_one_of(list(RDFList(self.g, lst)), name)
        return self.composite_datarange([], "opaque", name=name)

    #: XSD facets with an OCL reading, as ``(operator, operand form)``. ``size``
    #: facets compare ``v.size()``; the others compare ``v`` itself.
    _FACET_OPS = {
        "minInclusive": (">=", False), "maxInclusive": ("<=", False),
        "minExclusive": (">", False), "maxExclusive": ("<", False),
        "length": ("=", True), "minLength": (">=", True), "maxLength": ("<=", True),
    }

    def _facet_ocl(self, facet: str, node, var: str = "v") -> str | None:
        """OCL test of one XSD facet (O05) against ``var``, or ``None`` when it has no reading."""
        if facet == "pattern":
            return f"{var}.matches({naming.regex_literal(str(node))})"
        spec = self._FACET_OPS.get(facet)
        if spec is None:
            return None
        op, on_size = spec
        subject = f"{var}.size()" if on_size else var
        return f"{subject} {op} {self.ocl_literal(node)}"

    def _restricted_datatype(self, node, base, name: str | None) -> str:
        """O05: ``owl:onDatatype`` + ``owl:withRestrictions`` -> a datatype carrying the facets as an invariant."""
        base_name = self.resolve_data_range(base)
        facets: list[str] = []
        tests: list[str] = []
        restr = self.g.value(node, OWL.withRestrictions)
        if restr is not None:
            for f in RDFList(self.g, restr):
                for fp, fo in sorted(self.g.predicate_objects(f), key=lambda t: (str(t[0]), str(t[1]))):
                    local = naming.local_name(fp)
                    facets.append(f"{local} {self.literal_value(fo)}")
                    test = self._facet_ocl(local, fo)
                    if test:
                        tests.append(test)
                    else:
                        self.warn("DATATYPE_FACET_UNSUPPORTED",
                                  f"Facet '{local}' on a restriction of {base_name} has no OCL equivalent; ignored.")
        if name is None:
            key = ("restriction", base_name, tuple(facets))
            cached = self._composite_datarange_cache.get(key)
            if cached:
                return cached
            name = self.canon.reserve(f"_{base_name}_restricted")
            self._composite_datarange_cache[key] = name
        dt = PlannedDataType(name=name, facets=facets)
        # The facets constrain the datatype's own value, as for O01-O03.
        value_type = base_name if base_name in _PRIMITIVES else "any"
        dt.attributes.append(PlannedAttribute(name="value", type=value_type, lower=0, upper="*"))
        if tests:
            dt.invariants.append(PlannedInvariant(
                context=name, body=f"self.value->forAll(v | {' and '.join(tests)})",
                name="facet_invariant", origin_rule="O05",
            ))
        self.model.datatypes[name] = dt
        return name

    def _data_one_of(self, literals: list, name: str | None) -> str:
        """D17 / O04: ``owl:oneOf`` over literals.

        Plain and ``xsd:string`` literals become an enumeration (D17). Any other
        literal type cannot be an enumeration literal name, so it becomes a
        datatype whose ``value`` is restricted to the listed values (O04).
        """
        literals = [lit for lit in literals if isinstance(lit, Literal)]
        if not literals:
            return self.composite_datarange([], "opaque", name=name)
        if all(lit.datatype in (None, XSD.string, RDF.langString) for lit in literals):
            hint = naming.one_of_name([naming.sanitize(str(lit)) for lit in literals])
            return self.make_enum([self.literal_value(lit) for lit in literals], hint=hint, exact_name=name)
        types = {primitive_for(lit.datatype) for lit in literals}
        value_type = types.pop() if len(types) == 1 else "any"
        key = ("oneOf", tuple(sorted((str(lit), str(lit.datatype)) for lit in literals)))
        if name is None:
            cached = self._composite_datarange_cache.get(key)
            if cached:
                return cached
            name = self.canon.reserve(naming.one_of_name([naming.sanitize(str(lit)) for lit in literals]))
            self._composite_datarange_cache[key] = name
        dt = PlannedDataType(name=name)
        dt.attributes.append(PlannedAttribute(name="value", type=value_type, lower=0, upper="*"))
        choices = " or ".join(f"v = {self.ocl_literal(lit)}" for lit in literals)
        dt.invariants.append(PlannedInvariant(
            context=name, body=f"self.value->forAll(v | {choices})",
            name="oneOf_invariant", origin_rule="O04",
        ))
        self.model.datatypes[name] = dt
        return name

    def composite_datarange(self, members: list[str], kind: str, name: str | None = None) -> str:
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

        A ``name`` makes it a named datatype definition (D37): the type is built
        under that name and bypasses the structural cache.
        """
        # BUML's datatype vocabulary is coarser than the paper's (anyURI,
        # gYear and gYearMonth all collapse onto ``str``), so an OWL union of
        # four XSD types can arrive here with repeated members. Deduplicate
        # before building the name and the OCL body, otherwise the invariant
        # reads "... or v.oclIsTypeOf(str) or v.oclIsTypeOf(str)".
        members = list(dict.fromkeys(members))
        key = (kind, tuple(sorted(set(members))))
        if name is None:
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
            self._composite_datarange_cache[key] = name
        dt = PlannedDataType(name=name)
        if kind in ("union", "intersection", "complement") and members:
            # "any" (lowercase) is the BUML primitive; see ``axioms.primitive_for``.
            dt.attributes.append(PlannedAttribute(name="value", type="any", lower=0, upper="*"))
            if kind == "union":
                expr = " or ".join(f"v.oclIsTypeOf({m})" for m in members)
                origin = "O02"
            elif kind == "intersection":
                expr = " and ".join(f"v.oclIsTypeOf({m})" for m in members)
                origin = "O01"
            else:                                      # complement (single operand)
                expr = f"not v.oclIsTypeOf({members[0]})"
                origin = "O03"
            dt.invariants.append(PlannedInvariant(
                context=name, body=f"self.value->forAll(v | {expr})",
                name=f"{kind}_invariant", origin_rule=origin,
            ))
        self.model.datatypes[name] = dt
        return name

    def satisfies(self, dr: str, var: str) -> str:
        """Return the OCL test that ``var`` conforms to the data range ``dr``."""
        return f"{var}.oclIsKindOf({dr})"

    # =====================================================================
    #  phases
    # =====================================================================
    def _map_datatypes(self) -> None:
        for dt in self.ax.sorted_datatypes:
            if self.canon.is_alias(dt):
                continue
            self.resolve_data_range(dt)

    #: What a named class may say about itself and still be replaced by the
    #: enumeration it is equivalent to (D20): its definition, annotations, being
    #: a range, and having individuals.
    _ENUM_SUBJECT_PREDICATES = frozenset({
        RDF.type, RDFS.label, RDFS.comment, RDFS.isDefinedBy, OWL.deprecated,
        OWL.equivalentClass, RDFS.subClassOf,
    })
    _ENUM_OBJECT_PREDICATES = frozenset({RDF.type, RDFS.range})

    def _map_enum_classes(self) -> None:
        """D20: a named class that is exactly a ``oneOf`` of symbolic individuals *is* an enumeration.

        The enumeration takes the class's own name, and every property ranging
        over the class becomes an attribute typed by it. A class that also takes
        part in anything an enumeration cannot (a generalization, a domain, a
        restriction, a union operand, ...) stays a class, with a warning, and
        the ``oneOf`` is materialised beside it as before.
        """
        for c in self.ax.sorted_named_classes:
            if self.canon.is_alias(c):
                continue
            defs = [o for p, o in self.g.predicate_objects(c) if p in (OWL.equivalentClass, RDFS.subClassOf)]
            one_ofs = [o for o in defs if isinstance(o, BNode) and (o, OWL.oneOf, None) in self.g]
            if not one_ofs:
                continue
            if len(defs) != 1:
                self.warn("ONEOF_CLASS_KEPT",
                          f"Class '{naming.local_name(c)}' is a oneOf enumeration but also takes part in "
                          f"other axioms; kept as a class.")
                continue
            head = self.g.value(defs[0], OWL.oneOf)
            members = list(RDFList(self.g, head)) if head is not None else []
            if not members or any(not isinstance(m, Literal) and self.resolver._is_relational(m) for m in members):
                continue
            standalone = (
                all(p in self._ENUM_SUBJECT_PREDICATES for p in self.g.predicates(c, None))
                and all(p in self._ENUM_OBJECT_PREDICATES for _s, p in self.g.subject_predicates(c))
            )
            if not standalone:
                self.warn("ONEOF_CLASS_KEPT",
                          f"Class '{naming.local_name(c)}' is a oneOf enumeration but also takes part in "
                          f"other axioms; kept as a class.")
                continue
            self._enum_defs[c] = defs[0]
            name = self.canon.name(c)
            labels = [self.literal_value(m) if isinstance(m, Literal) else naming.sanitize(naming.local_name(m))
                      for m in members]
            self.make_enum(labels, hint=name, exact_name=name)
            self.model.enumerations[name].uri = str(c)

    def _map_classes(self) -> None:
        for c in self.ax.sorted_named_classes:
            if c in self._enum_defs:
                continue
            self.ensure_class(c)

    def _map_class_axioms(self) -> None:
        # D26 subClassOf (named or expression); restriction/union superclasses via resolver
        for sub, sup in self.ax.subclass_axioms():
            if self._enum_defs.get(sub) == sup:
                continue                           # D20: the class is the enumeration
            sub_name = self.ensure_class(sub)
            if isinstance(sup, URIRef):
                self.add_generalization(sub_name, self.ensure_class(sup))
            else:
                self.resolver.resolve_class(sup, subsuming=sub_name)
        # equivalentClass to an anonymous expression -> treat like subclass wiring
        for s, o in self.ax.equivalent_classes():
            if self._enum_defs.get(s) == o or s in self.ax.datatypes():
                continue                           # D20 / D37: an enumeration or a datatype definition
            if isinstance(s, URIRef) and isinstance(o, BNode):
                self.resolver.resolve_class(o, subsuming=self.ensure_class(s))
        # O16 DisjointClasses (pairwise OCL)
        for a, b in _sorted_pairs(self.g.subject_objects(OWL.disjointWith)):
            if isinstance(a, URIRef) and isinstance(b, URIRef):
                self.add_ocl(self.ensure_class(a),
                             f"not self.oclIsKindOf({self.ensure_class(b)})", origin="O16")
        # O17 DisjointUnion
        for c, lst in _sorted_pairs(self.g.subject_objects(OWL.disjointUnionOf)):
            members = [self.resolver.resolve_class(m) for m in RDFList(self.g, lst)]
            cname = self.ensure_class(c)
            for m in members:
                self.add_generalization(m, cname)
            disj = " or ".join(f"self.oclIsKindOf({m})" for m in members)
            self.add_ocl(cname, disj, origin="O17", name="complete")

    def _combine(self, nodes) -> str:
        """Resolve one-or-more domain/range nodes to a single class name.

        The single place a property's ``rdfs:domain`` / ``rdfs:range`` becomes a
        class, for object and data properties alike. Class expressions are
        resolved bottom-up first (:class:`~owl2uml.expressions.ExpressionResolver`),
        so a union domain lands on the auxiliary union class D19 materialised for
        it — the one every member already generalises to — rather than being
        expanded back into its members. Several ``rdfs:domain`` axioms mean the
        intersection of them, and an absent domain/range falls back to Thing.

        Assigning the property to the resolved class is what keeps the OCL
        emitted for it navigable: every rule that references ``self.<p>`` picks
        its context through this same method, so the feature is always on the
        context class or one of its ancestors.
        """
        names = sorted({self.resolver.resolve_class(n) for n in nodes})
        if not names:
            return self.ensure_class(OWL.Thing)
        if len(names) == 1:
            return names[0]
        aux = self.make_aux_class(naming.intersection_name(names))
        for nm in names:
            self.add_generalization(aux, nm)
        return aux

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
        """D10, D29-D32, O22 — one association per object property.

        D30/D31 resolve a union-typed domain/range to the union class itself
        (see :meth:`_combine`) rather than linking to each member separately.
        Linking to the members put the association *below* every invariant O18 /
        O21 / O23-O26 emits for the same property, whose context is that same
        resolved class, so ``self.<p>`` was unresolvable for any OCL evaluator:
        they walk a context's ancestors, never its subclasses.
        """
        # D29 inverse pairs first (one bidirectional association per pair)
        for a, b in self.ax.inverse_pairs():
            if not (isinstance(a, URIRef) and isinstance(b, URIRef)):
                continue
            if a in self._handled_props or b in self._handled_props:
                continue
            src = self._combine(self._group_domains(a) or self._group_ranges(b))
            tgt = self._combine(self._group_ranges(a) or self._group_domains(b))
            if src in self.model.enumerations or tgt in self.model.enumerations:
                continue                          # an enumeration has no end to navigate back from
            self._handled_props.update({a, b})
            aname, bname = self.canon.name(a), self.canon.name(b)
            # D32 / O22 for both halves of the pair: a functional property bounds
            # the end it points at, an inverse-functional one bounds the end it
            # starts from, and each is the other's mirror image.
            tgt_mult = (0, 1) if self.ax.is_functional(a) or self.ax.is_inverse_functional(b) else (0, "*")
            src_mult = (0, 1) if self.ax.is_functional(b) or self.ax.is_inverse_functional(a) else (0, "*")
            self.add_association(src, tgt, role=aname, source_role=bname,
                                 src_mult=src_mult, tgt_mult=tgt_mult, uri=str(a))
            if self.ax.is_inverse_functional(a):     # O22
                self._add_inverse_functional_ocl(src, tgt, aname)
            if self.ax.is_inverse_functional(b):     # O22, from the other side
                self._add_inverse_functional_ocl(tgt, src, bname)
        # remaining object properties
        for p in self.ax.sorted_object_props:
            if self.canon.is_alias(p) or p in self._handled_props:
                continue
            self._warn_missing_domain_range(p, self._group_domains(p), self._group_ranges(p))
            src = self._combine(self._group_domains(p))
            tgt = self._combine(self._group_ranges(p))
            tgt_mult = (0, 1) if self.ax.is_functional(p) else (0, "*")
            src_mult = (0, 1) if self.ax.is_inverse_functional(p) else (0, "*")
            pname = self.canon.name(p)
            self.add_feature(src, tgt, pname, src_mult=src_mult, tgt_mult=tgt_mult, uri=str(p))
            if self.ax.is_inverse_functional(p) and tgt not in self.model.enumerations:     # O22
                self._add_inverse_functional_ocl(src, tgt, pname)

    def _add_inverse_functional_ocl(self, src: str, tgt: str, role: str) -> None:
        """O22: no ``tgt`` instance is the ``role`` of more than one ``src`` instance."""
        self.add_ocl(tgt, f"{tgt}.allInstances()->forAll(d | {src}.allInstances()"
                          f"->select(c | c.{role}->includes(d))->size() <= 1)",
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
            cls.attributes.append(PlannedAttribute(name=pname, type=dtype, lower=0, upper=upper,
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

        ``Thing`` is checked alongside the walked generalization chain because a
        domain-less property falls back to it (see ``_map_data_properties`` /
        ``_map_object_properties``), so the property does exist in the model.
        Whether the context can actually *navigate* it is a different question,
        answered against the finished model by ``to_buml._parses_against_model``:
        Thing is only an ancestor if the user accepted the ``attach_to_thing``
        recommendation for it (see ``_complete_thing_hierarchy``).
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
                if assoc.source == name and assoc.role == pname:
                    return True
                if (assoc.target == name and assoc.navigable_source
                        and assoc.source_role == pname):
                    return True
            for sub, sup in self.model.generalizations:
                if sub == name:
                    stack.append(sup)
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
        self._derive_data_superproperties()

    def _derive_data_superproperties(self) -> None:
        """O27: a data property that only its sub-properties feed is their union.

        Needs two or more sub-properties of one super-property and no assertion
        on the super-property itself, the same "exclusively derived" test as O19.
        With a single sub-property the inclusion O27 already emitted is all
        there is to say.
        """
        subs_of: dict = {}
        for sub, sup in self.ax.sub_property_axioms():
            if self.ax.is_data_property(sub) and sub != sup and self.canon.name(sub) != self.canon.name(sup):
                subs_of.setdefault(sup, []).append(sub)
        for sup, subs in sorted(subs_of.items(), key=lambda t: str(t[0])):
            subs = sorted({self.canon.name(x) for x in subs})
            supn = self.canon.name(sup)
            if len(subs) < 2 or self.ax.has_assertions(sup):
                continue
            domains = self._group_domains(sup)
            ctx = self._combine(domains) if domains else self.ensure_class(OWL.Thing)
            missing = [n for n in [supn, *subs] if not self._has_property(ctx, n)]
            if missing:
                self.warn("DERIVATION_SKIPPED",
                          f"Cannot derive '{supn}' from its sub-properties: {', '.join(missing)} "
                          f"is not available on {ctx}.")
                continue
            union = "->union(".join(f"self.{n}->asSet()" for n in subs) + ")" * (len(subs) - 1)
            self.add_ocl(ctx, f"self.{supn}->asSet() = {union}", origin="O27", name="derived")
            for attr in self.model.classes[ctx].attributes:
                if attr.name == supn:
                    attr.is_derived = True

    def _map_property_chains(self) -> None:
        """O19 / O20: ``owl:propertyChainAxiom``.

        A property that no individual asserts directly is *exclusively derived*
        by its chain (O19): its association is marked derived and the chain
        both implies and is implied by it. Otherwise the chain only implies the
        property (O20). B-OCL has no ``def`` form, so the derivation is stated
        as an invariant rather than as ``/OPE : Set(T) = ...``.
        """
        for p, chain in self.ax.property_chains():
            if self.canon.is_alias(p):
                continue
            pn = self.canon.name(p)
            names = [self.canon.name(c) for c in chain]
            domains = self._group_domains(p) or self._group_domains(chain[0])
            ctx = self._combine(domains) if domains else self.ensure_class(OWL.Thing)
            if not self._has_property(ctx, pn) or not self._has_property(ctx, names[0]):
                self.warn("CHAIN_SKIPPED",
                          f"Property chain for '{pn}' skipped: '{pn}' or '{names[0]}' is not available on {ctx}.")
                continue
            walk = "self." + ".".join(names)
            body = f"{walk}->forAll(o | self.{pn}->includes(o))"
            if self.ax.has_assertions(p):                                         # O20
                self.add_ocl(ctx, body, origin="O20", name="chain")
                continue
            body += f" and self.{pn}->forAll(o | {walk}->includes(o))"          # O19
            self.add_ocl(ctx, body, origin="O19", name="derived")
            for assoc in self.model.associations:
                if assoc.uri == str(p):
                    assoc.derived = True

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
        # O21 / O28 DisjointObjectProperties / DisjointDataProperties (pairwise)
        pairs = {(a, b) for a, b in self.g.subject_objects(OWL.propertyDisjointWith)}
        for members in self.ax.all_disjoint_properties():       # the n-ary form is every pair
            pairs.update((a, b) for i, a in enumerate(members) for b in members[i + 1:])
        for a, b in sorted(pairs, key=lambda t: (str(t[0]), str(t[1]))):
            if not (isinstance(a, URIRef) and isinstance(b, URIRef)):
                continue
            an, bn = self.canon.name(a), self.canon.name(b)
            ctx = self._combine(self._group_domains(a)) if self._group_domains(a) \
                else self.ensure_class(OWL.Thing)
            if self.ax.is_object_property(a) and self.ax.is_object_property(b):
                self.add_ocl(ctx, f"self.{an}->forAll(o | self.{bn}->excludes(o))", origin="O21")
            elif self.ax.is_data_property(a) and self.ax.is_data_property(b):
                self.add_ocl(ctx, f"self.{an}->asSet()->intersection(self.{bn}->asSet())->isEmpty()", origin="O28")
            else:
                # OWL 2 only defines disjointness between two object or two
                # data properties; a mixed pair (or an undeclared property) has
                # no O21/O28 reading.
                self.warn("DISJOINT_PROPERTIES_SKIPPED",
                          f"'{an}' and '{bn}' are declared disjoint but are not both object "
                          f"properties or both data properties; no invariant emitted.")

    def _map_keys(self) -> None:
        """D38: ``owl:hasKey`` -> a uniqueness invariant, plus an identifier for a single functional key.

        The invariant says no two instances share all their key values. A key
        that is a single single-valued attribute of the class itself is also
        marked ``is_id``, which is what generators read as the primary key; a
        class may have only one, so the first one wins.
        """
        for cls_iri, keys in self.ax.has_keys():
            if self.canon.is_alias(cls_iri):
                continue
            cname = self.ensure_class(cls_iri)
            names = [self.canon.name(k) for k in keys]
            if cname not in self.model.classes:
                self.warn("KEY_SKIPPED",
                          f"Key of '{cname}' skipped: '{cname}' is not modelled as a class.")
                continue
            missing = [n for n in names if not self._has_property(cname, n)]
            if missing:
                self.warn("KEY_SKIPPED",
                          f"Key of '{cname}' skipped: {', '.join(missing)} "
                          f"{'is' if len(missing) == 1 else 'are'} not modelled as a feature of it.")
                continue
            if len(names) == 1:
                body = f"{cname}.allInstances()->isUnique({names[0]})"
            else:
                differ = " or ".join(f"a.{n} <> b.{n}" for n in names)
                body = (f"{cname}.allInstances()->forAll(a | {cname}.allInstances()"
                        f"->forAll(b | a <> b implies ({differ})))")
            self.add_ocl(cname, body, origin="D38", name="key")
            cls = self.model.classes[cname]
            attr = next((a for a in cls.attributes if len(names) == 1 and a.name == names[0]), None)
            if attr is not None and attr.upper == 1 and not any(a.is_id for a in cls.attributes):
                attr.is_id = True

    def _map_individuals(self) -> None:
        for i in self.ax.sorted_individuals:
            self.ensure_instance(i)
            for p, o in self.ax.object_assertions(i):       # D40
                self.ensure_instance(o)
                self.canon.name(p)                          # reserve the role's name

    def _post_checks(self) -> None:
        """Create stubs for any type referenced but not yet declared."""
        known = self.model.all_type_names()
        referenced: set[str] = set()
        for a in self.model.associations:
            referenced.update({a.source, a.target})
        for cls in self.model.classes.values():
            for attr in cls.attributes:
                referenced.add(attr.type)
        for pair in self.model.generalizations:
            referenced.update(pair)
        for missing in sorted(referenced - known - _PRIMITIVES):
            if missing and missing not in self.model.classes:
                self.model.classes[missing] = PlannedClass(name=missing, is_stub=True)
