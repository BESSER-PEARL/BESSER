"""
Stress-test the OCL method-contract pipeline (PR #529 consolidation).

Generates ~400 distinct pre/post constraints across many class+method
shapes and verifies:
  * routing into Method.pre / Method.post (NOT domain_model.constraints)
  * canonical full-text expression on `.expression`
  * BUML round-trip via domain_model_to_code preserves pre/post lines
  * Re-import of generated BUML preserves routing and expression text

Run: python _stress_ocl_method_contracts.py
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code

random.seed(42)

# NOTE: B-OCL grammar accepts int/float/bool/str as param types but does not
# recognise `date` in a method header (parser raises "Unexpected token near
# 'date'"). We exclude it from the main pool and exercise it only in a
# dedicated expected-grammar-failure bucket so the main routing/canonical
# numbers measure what we actually care about (PR #529 routing/round-trip).
PRIM_TYPES = ["int", "float", "bool", "str"]
# These bodies depend on what's in scope. We keep bodies generic so they parse
# under most signatures -- using 'self.' refs (resolved against attributes) and
# parameter-only refs.
SELF_BODIES = [
    "self.balance >= 0",
    "self.x > 0",
    "self.is_active = true",
    "self.balance > 10",
]
PARAM_BODIES_BY_TYPE = {
    "int":   ["{p} > 0",  "{p} >= 0",  "{p} < 1000"],
    "float": ["{p} > 0",  "{p} >= 0.0"],
    "bool":  ["{p} = true", "{p} = false"],
    "str":   ["{p} <> ''"],
}


@dataclass
class CaseResult:
    case_id: str
    expected_kind: str  # "pre" / "post" / "expected_failure_unknown" / "expected_failure_named_pre" / "expected_failure_dup"
    routing_ok: bool = False
    canonical_ok: bool = False
    buml_roundtrip_ok: bool = False
    failure_mode: Optional[str] = None
    failure_text: Optional[str] = None


@dataclass
class Stats:
    total: int = 0
    routing_pass: int = 0
    canonical_pass: int = 0
    buml_pass: int = 0
    expected_failures_total: int = 0
    expected_failures_correct: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (text, error)


def _build_method_signature(name: str, params: list[tuple[str, str]], rtype: str = "int") -> str:
    sig = ", ".join(f"{p}: {t}" for p, t in params)
    return f"+ {name}({sig}): {rtype}"


def _build_class_diagram(
    classes: list[dict],
    constraints: list[dict],
    title: str = "StressTest",
) -> dict:
    """Build a JSON ClassDiagram payload.

    classes: list of {name, attributes:[(name,type)], methods:[(name,params,rtype)]}
    constraints: list of {target_class_name, text}
    """
    elements: dict[str, dict] = {}
    relationships: dict[str, dict] = {}

    class_id_by_name: dict[str, str] = {}

    for cls in classes:
        cls_id = f"cls-{cls['name']}-{uuid.uuid4().hex[:8]}"
        class_id_by_name[cls['name']] = cls_id
        attr_ids: list[str] = []
        method_ids: list[str] = []

        for attr_name, attr_type in cls.get('attributes', []):
            aid = f"attr-{cls['name']}-{attr_name}-{uuid.uuid4().hex[:6]}"
            elements[aid] = {
                "id": aid,
                "name": attr_name,
                "visibility": "public",
                "attributeType": attr_type,
            }
            attr_ids.append(aid)

        for m_name, m_params, m_rtype in cls.get('methods', []):
            mid = f"m-{cls['name']}-{m_name}-{uuid.uuid4().hex[:6]}"
            elements[mid] = {
                "id": mid,
                "name": _build_method_signature(m_name, m_params, m_rtype),
                "type": "ClassMethod",
                "owner": cls_id,
            }
            method_ids.append(mid)

        elements[cls_id] = {
            "id": cls_id,
            "name": cls['name'],
            "type": "Class",
            "attributes": attr_ids,
            "methods": method_ids,
            "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
        }

    for con in constraints:
        oid = f"ocl-{uuid.uuid4().hex[:10]}"
        elements[oid] = {
            "id": oid,
            "type": "ClassOCLConstraint",
            "constraint": con['text'],
        }
        target_cls_id = class_id_by_name[con['target_class_name']]
        rid = f"r-{uuid.uuid4().hex[:8]}"
        relationships[rid] = {
            "id": rid,
            "type": "ClassOCLLink",
            "source": {"element": oid},
            "target": {"element": target_cls_id},
        }

    return {
        "title": title,
        "model": {"elements": elements, "relationships": relationships},
    }


def _gen_classes_and_cases() -> tuple[list[dict], list[dict], list[CaseResult]]:
    """Build many classes & methods, plus matching constraint cases."""
    classes: list[dict] = []
    constraints: list[dict] = []
    cases: list[CaseResult] = []

    # 30 classes; each 1-5 methods; each method 0/1/2/4 params; mix of types.
    # Some methods get pre, some post, some both; many bodies use self. or params.
    n_classes = 30
    method_count_choices = [1, 2, 3, 4, 5]
    param_count_choices = [0, 1, 2, 4]

    # We'll target ~400 main constraints
    total_target = 400
    produced = 0

    forward_ref_class_names = [f"Cls{i}" for i in range(n_classes)]

    for i in range(n_classes):
        cls_name = f"Cls{i}"
        # Standard attributes that constraint bodies can reference
        attrs = [("balance", "int"), ("x", "int"), ("is_active", "bool")]
        n_methods = random.choice(method_count_choices)
        methods = []
        for j in range(n_methods):
            m_name = f"op{j}"
            n_params = random.choice(param_count_choices)
            params = []
            for k in range(n_params):
                # ~1/6 of params use a class type forward-ref
                if random.random() < 0.15 and forward_ref_class_names:
                    p_type = random.choice([c for c in forward_ref_class_names if c != cls_name] or [cls_name])
                else:
                    p_type = random.choice(PRIM_TYPES)
                params.append((f"p{k}", p_type))
            methods.append((m_name, params, "int"))
        classes.append({"name": cls_name, "attributes": attrs, "methods": methods})

    # Now generate constraints per method. We aim for ~ total_target valid pre/post.
    # Walk the classes/methods; for each method, generate a random mix.
    method_inventory = []  # list of (cls_name, m_name, params)
    for cls in classes:
        for (m_name, params, _rt) in cls['methods']:
            method_inventory.append((cls['name'], m_name, params))

    # Normal valid pre/post cases
    while produced < total_target:
        cls_name, m_name, params = random.choice(method_inventory)
        sig = ", ".join(f"{p}: {t}" for p, t in params)
        kind = random.choice(["pre", "post"])
        # Choose body
        body_choices = list(SELF_BODIES)
        for (p, t) in params:
            for tmpl in PARAM_BODIES_BY_TYPE.get(t, []):
                # only use prim type bodies; skip class-typed params
                body_choices.append(tmpl.format(p=p))
        body = random.choice(body_choices)
        text = f"context {cls_name}::{m_name}({sig}) {kind}: {body}"
        case_id = f"valid-{produced}"
        constraints.append({"target_class_name": cls_name, "text": text})
        cases.append(CaseResult(case_id=case_id, expected_kind=kind))
        produced += 1

    # Edge case: `pre name: body` -- BOCL grammar rejects it (only invariants take name).
    # Pick 8 such cases.
    for n in range(8):
        cls_name, m_name, params = random.choice(method_inventory)
        sig = ", ".join(f"{p}: {t}" for p, t in params)
        text = f"context {cls_name}::{m_name}({sig}) pre myname{n}: self.balance > 0"
        case_id = f"named-pre-{n}"
        constraints.append({"target_class_name": cls_name, "text": text})
        cases.append(CaseResult(case_id=case_id, expected_kind="expected_failure_named_pre"))

    # Edge case: typo'd method name
    for n in range(8):
        cls_name, _m_name, _params = random.choice(method_inventory)
        text = f"context {cls_name}::not_a_method() pre: self.balance > 0"
        case_id = f"unknown-method-{n}"
        constraints.append({"target_class_name": cls_name, "text": text})
        cases.append(CaseResult(case_id=case_id, expected_kind="expected_failure_unknown"))

    # Edge case: B-OCL grammar limitation -- `date` as a param type in the
    # method header is rejected by the parser. We exercise this in a dedicated
    # bucket so we can verify the warning path works without polluting the main
    # routing-pass numbers.
    for n in range(8):
        cls_name, m_name, _ = random.choice(method_inventory)
        text = f"context {cls_name}::{m_name}(d: date) pre: self.balance >= 0"
        case_id = f"date-grammar-{n}"
        constraints.append({"target_class_name": cls_name, "text": text})
        cases.append(CaseResult(case_id=case_id, expected_kind="expected_failure_grammar"))

    # Edge case: same method name in two different classes; constraint applied separately to each.
    # Add a `shared_op` to two classes:
    classes[0]['methods'] = list(classes[0]['methods']) + [("shared_op", [("q", "int")], "int")]
    classes[1]['methods'] = list(classes[1]['methods']) + [("shared_op", [("q", "int")], "int")]
    text_a = f"context {classes[0]['name']}::shared_op(q: int) pre: q > 0"
    text_b = f"context {classes[1]['name']}::shared_op(q: int) pre: q > 5"
    constraints.append({"target_class_name": classes[0]['name'], "text": text_a})
    cases.append(CaseResult(case_id="shared-A", expected_kind="pre"))
    constraints.append({"target_class_name": classes[1]['name'], "text": text_b})
    cases.append(CaseResult(case_id="shared-B", expected_kind="pre"))

    return classes, constraints, cases


def _find_method(dm, cls_name: str, m_name: str):
    for t in dm.types:
        if hasattr(t, 'methods') and getattr(t, 'name', None) == cls_name:
            for m in (t.methods or []):
                if m.name == m_name:
                    return t, m
    return None, None


def main():
    t0 = time.time()
    stats = Stats()

    classes, constraints, cases = _gen_classes_and_cases()
    payload = _build_class_diagram(classes, constraints)

    # ---- Pass 1: ingest & route ----
    dm = process_class_diagram(payload)
    warnings = list(getattr(dm, 'ocl_warnings', []) or [])

    # Map: (cls_name, m_name) -> list of (kind, expression)
    method_routing: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for t in dm.types:
        if not hasattr(t, 'methods'):
            continue
        for m in (t.methods or []):
            for c in (m.pre or []):
                method_routing.setdefault((t.name, m.name), []).append(("pre", c.expression))
            for c in (m.post or []):
                method_routing.setdefault((t.name, m.name), []).append(("post", c.expression))

    # Quick check: invariants on dm.constraints (should be empty for our test)
    bad_inv_routing = len(dm.constraints or []) > 0

    # We need to count successes per case. For each `valid-*` case we look up
    # whether the exact expression text is on the right method's pre/post list.
    valid_cases = [c for c in cases if c.expected_kind in ("pre", "post")]
    ef_named_pre = [c for c in cases if c.expected_kind == "expected_failure_named_pre"]
    ef_unknown = [c for c in cases if c.expected_kind == "expected_failure_unknown"]
    ef_grammar = [c for c in cases if c.expected_kind == "expected_failure_grammar"]

    # Build a quick search of all expected expressions
    constraints_by_text = {c['text']: c for c in constraints}

    # For counting we iterate cases in order paired with constraints (1:1 above
    # for non-shared cases). We rebuild the pairing:
    case_pairs = list(zip(cases, constraints))

    for case, con in case_pairs:
        if case.expected_kind in ("pre", "post"):
            stats.total += 1
            # Find method header: parse "context X::m(... ) kind: body" -> X, m
            text = con['text']
            # cheap parse:
            try:
                hdr = text.split("context ", 1)[1]
                cls_name = hdr.split("::", 1)[0].strip()
                m_name = hdr.split("::", 1)[1].split("(", 1)[0].strip()
            except Exception:
                cls_name = con['target_class_name']
                m_name = "?"
            entries = method_routing.get((cls_name, m_name), [])
            if any(k == case.expected_kind and e == text for (k, e) in entries):
                case.routing_ok = True
                case.canonical_ok = True
                stats.routing_pass += 1
                stats.canonical_pass += 1
            elif any(e == text for (_k, e) in entries):
                # canonical text right but kind mismatch
                case.canonical_ok = True
                stats.canonical_pass += 1
                case.failure_mode = "wrong-kind"
                if len(stats.failures) < 5:
                    stats.failures.append((text, "wrong-kind routing"))
            else:
                # Did the expression land but body-only?
                # Check whether some body-only string equals just the body
                body = text.split(":", 2)[-1].strip()
                if any(e == body for (_k, e) in entries):
                    case.failure_mode = "body-only-expression"
                    if len(stats.failures) < 5:
                        stats.failures.append((text, "expression is body-only, not canonical"))
                else:
                    case.failure_mode = "not-routed"
                    # try to surface a warning that mentions the method
                    related = next((w for w in warnings if m_name in w and cls_name in w), None)
                    if len(stats.failures) < 5:
                        stats.failures.append((text, related or "no warning found; missing entirely"))

    # Expected failures: named-pre and unknown-method should NOT route
    # (no entry on pre/post) and SHOULD produce a warning.
    def _is_routed(text: str) -> bool:
        try:
            hdr = text.split("context ", 1)[1]
            cls_name = hdr.split("::", 1)[0].strip()
            m_name = hdr.split("::", 1)[1].split("(", 1)[0].strip()
        except Exception:
            return False
        return any(e == text for (_k, e) in method_routing.get((cls_name, m_name), []))

    for c in ef_named_pre + ef_unknown + ef_grammar:
        stats.expected_failures_total += 1
    for case, con in case_pairs:
        if case.expected_kind not in (
            "expected_failure_named_pre",
            "expected_failure_unknown",
            "expected_failure_grammar",
        ):
            continue
        if not _is_routed(con['text']):
            stats.expected_failures_correct += 1
        else:
            if len(stats.failures) < 5:
                stats.failures.append((con['text'], f"expected-failure case unexpectedly routed"))

    # ---- Pass 2: BUML round-trip ----
    buml_pass_count = 0
    buml_failure_recorded = False
    re_import_failure_recorded = False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "stress_dm.py")
        domain_model_to_code(dm, file_path=tmp_file)
        with open(tmp_file, "r", encoding="utf-8") as f:
            generated = f.read()

        # Count how many "<cls>_m_<m>.add_pre" or ".add_post" lines exist
        # and ensure each routed pre/post got one.
        for case, con in case_pairs:
            if case.expected_kind not in ("pre", "post"):
                continue
            text = con['text']
            try:
                hdr = text.split("context ", 1)[1]
                cls_name = hdr.split("::", 1)[0].strip()
                m_name = hdr.split("::", 1)[1].split("(", 1)[0].strip()
            except Exception:
                continue
            kind = case.expected_kind
            needle = f"{cls_name}_m_{m_name}.add_{kind}("
            if needle in generated:
                # The expression is written through Python string-literal
                # escaping (`_escape_python_string`), so a raw `text in
                # generated` substring search would miss perfectly valid
                # outputs whenever the body contains `'`, `"`, `\\`, etc.
                # We treat the presence of the `add_<kind>(...)` line as the
                # syntactic round-trip signal, then defer the semantic check
                # to the re-import pass below (which evaluates the literal
                # back to a string and compares it against `text`).
                case.buml_roundtrip_ok = True
                buml_pass_count += 1
            else:
                if not buml_failure_recorded and len(stats.failures) < 5:
                    stats.failures.append((text, f"BUML missing line: {needle}"))
                    buml_failure_recorded = True

        stats.buml_pass = buml_pass_count

        # Re-import the generated module, fetch domain_model, verify routing.
        spec = importlib.util.spec_from_file_location("stress_dm_reimport", tmp_file)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            dm2 = getattr(mod, "domain_model")
            # Build the same routing index
            method_routing2: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for t in dm2.types:
                if not hasattr(t, 'methods'):
                    continue
                for m in (t.methods or []):
                    for cn in (m.pre or []):
                        method_routing2.setdefault((t.name, m.name), []).append(("pre", cn.expression))
                    for cn in (m.post or []):
                        method_routing2.setdefault((t.name, m.name), []).append(("post", cn.expression))
            reimport_routing_pass = 0
            reimport_total = 0
            for case, con in case_pairs:
                if case.expected_kind not in ("pre", "post"):
                    continue
                reimport_total += 1
                text = con['text']
                try:
                    hdr = text.split("context ", 1)[1]
                    cls_name = hdr.split("::", 1)[0].strip()
                    m_name = hdr.split("::", 1)[1].split("(", 1)[0].strip()
                except Exception:
                    continue
                entries = method_routing2.get((cls_name, m_name), [])
                if any(k == case.expected_kind and e == text for (k, e) in entries):
                    reimport_routing_pass += 1
                else:
                    if not re_import_failure_recorded and len(stats.failures) < 5:
                        stats.failures.append((text, "re-import lost routing"))
                        re_import_failure_recorded = True
        except Exception as e:
            reimport_routing_pass = 0
            reimport_total = len([c for c in cases if c.expected_kind in ("pre", "post")])
            if len(stats.failures) < 5:
                stats.failures.append(("(re-import failure)", str(e)))

    elapsed = time.time() - t0

    # Final report
    valid_total = sum(1 for c in cases if c.expected_kind in ("pre", "post"))
    print("=" * 70)
    print("BESSER OCL Method-Contract Stress Test (PR #529)")
    print("=" * 70)
    print(f"Total constraints exercised:         {len(cases)}")
    print(f"  Valid pre/post cases:              {valid_total}")
    print(f"  Expected-failure cases:            {stats.expected_failures_total} "
          f"(named-pre + unknown-method)")
    print(f"Routing pass count:                  {stats.routing_pass} / {valid_total}")
    print(f"Canonical-expression pass count:     {stats.canonical_pass} / {valid_total}")
    print(f"BUML round-trip pass count:          {stats.buml_pass} / {valid_total}")
    print(f"BUML re-import routing preserved:    {reimport_routing_pass} / {reimport_total}")
    print(f"Expected-failure correctness:        "
          f"{stats.expected_failures_correct} / {stats.expected_failures_total}")
    print(f"Invariants leaked to dm.constraints: {'YES (BUG)' if bad_inv_routing else 'no'}")
    print(f"Total warnings emitted:              {len(warnings)}")
    print(f"Time taken:                          {elapsed:.2f}s")

    if stats.failures:
        print("-" * 70)
        print("Distinct failure modes (up to 5):")
        for i, (txt, err) in enumerate(stats.failures, 1):
            print(f"  [{i}] text: {txt[:100]}")
            print(f"      err : {err[:200]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
