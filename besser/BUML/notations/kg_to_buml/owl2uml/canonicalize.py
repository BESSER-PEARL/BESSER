"""Canonicalisation of equivalent classes / properties (D27, D28, D33) and the
IRI -> UML-name resolver (with cross-namespace collision disambiguation).

``owl:equivalentClass`` and ``owl:equivalentProperty`` merge several IRIs into a
single UML element.  A union-find groups them; one *representative* IRI is
chosen per group and every reference is rewritten to it.
"""
from __future__ import annotations

from rdflib import Graph, URIRef
from rdflib.namespace import OWL

from .naming import local_name, sanitize, namespace_hint


class Canonicalizer:
    def __init__(self, graph: Graph, base: str | None):
        self.graph = graph
        self.base = base or ""
        self._parent: dict[str, str] = {}
        self._name_cache: dict[str, str] = {}   # canonical iri -> UML name
        self._used_names: set[str] = set()
        self._build_unionfind()

    # ---- union-find ------------------------------------------------------
    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    def _build_unionfind(self) -> None:
        # Sorted so the union-find's parent chains — and therefore the
        # representative chosen for each equivalence group — do not depend on
        # rdflib's hash ordering.
        for pred in (OWL.equivalentClass, OWL.equivalentProperty):
            pairs = sorted(self.graph.subject_objects(pred), key=lambda t: (str(t[0]), str(t[1])))
            for s, o in pairs:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    self._union(str(s), str(o))

    # ---- representative choice ------------------------------------------
    def _describedness(self, iri: str) -> int:
        return len(list(self.graph.predicate_objects(URIRef(iri))))

    def rep(self, iri) -> str:
        """Return the canonical representative IRI for ``iri``."""
        s = str(iri)
        root = self._find(s)
        members = [k for k in self._parent if self._find(k) == root]
        if not members:
            return s
        return self._choose(members)

    def _choose(self, members: list[str]) -> str:
        def key(m: str):
            in_base = 0 if (self.base and m.startswith(self.base)) else 1
            return (in_base, -self._describedness(m), m)
        return sorted(members, key=key)[0]

    def is_alias(self, iri) -> bool:
        """True if ``iri`` is folded into a different representative."""
        return str(iri) != self.rep(iri)

    def group(self, iri) -> list[str]:
        """All IRIs in the same equivalence component as ``iri``."""
        s = str(iri)
        if s not in self._parent:
            return [s]
        root = self._find(s)
        return sorted(k for k in self._parent if self._find(k) == root)

    # ---- naming ----------------------------------------------------------
    def register_primary(self, iris: list) -> None:
        """Pre-assign collision-free names for the primary named elements.

        Group canonical reps by their sanitised local name; any name claimed by
        more than one distinct rep is disambiguated for *all* claimants with a
        namespace hint, so the result is deterministic and symmetric.
        """
        reps = sorted({self.rep(i) for i in iris})
        by_local: dict[str, list[str]] = {}
        for rep in reps:
            by_local.setdefault(sanitize(local_name(rep)), []).append(rep)
        for base_name, claimants in sorted(by_local.items()):
            if len(claimants) == 1:
                self._assign(claimants[0], base_name)
            else:
                for rep in claimants:
                    self._assign(rep, sanitize(f"{base_name}_{namespace_hint(rep)}"))

    def _assign(self, iri: str, name: str) -> None:
        # guarantee global uniqueness (aux names may also be in play)
        final = name
        n = 2
        while final in self._used_names and self._name_cache.get(iri) != final:
            final = f"{name}_{n}"
            n += 1
        self._name_cache[iri] = final
        self._used_names.add(final)

    def name(self, iri) -> str:
        """UML name for ``iri`` (canonicalised, sanitised, disambiguated)."""
        rep = self.rep(iri)
        if rep in self._name_cache:
            return self._name_cache[rep]
        self._assign(rep, sanitize(local_name(rep)))
        return self._name_cache[rep]

    def reserve(self, name: str) -> str:
        """Reserve a (possibly aux) name, uniquifying against used names."""
        final = name
        n = 2
        while final in self._used_names:
            final = f"{name}_{n}"
            n += 1
        self._used_names.add(final)
        return final
