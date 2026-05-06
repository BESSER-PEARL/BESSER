"""OCL pipeline stress test — ~400 malformed/edge-case inputs.

Verifies that the json_to_buml -> ocl_checker pipeline degrades gracefully:
no uncaught exceptions; every malformed input shows up in dm.ocl_warnings
or in the validator's invalid_constraints list.
"""
from __future__ import annotations

import sys
import time
import traceback
import uuid

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)


# ---------------------------------------------------------------------------
# JSON payload builders
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())


def make_payload(
    constraint_text: str,
    *,
    class_name: str = "Book",
    attr_name: str = "pages",
    method_name: str = "borrow",
    extra_classes: list[tuple[str, str]] | None = None,
    legacy_kind: str | None = None,
    legacy_constraint_name: str | None = None,
    description: str | None = None,
    duplicate_in_second_box: bool = False,
) -> dict:
    """Build a minimal ClassDiagram payload with a class + OCL constraint box.

    `legacy_kind`: if set, emits the legacy body-only shape (constraint=body,
    sibling kind/constraintName/targetMethodId); otherwise, the canonical
    full-text shape (constraint=full text).
    """
    class_id = _new_id()
    method_id = _new_id()
    attr_id = _new_id()
    constraint_id = _new_id()
    link_id = _new_id()

    elements = {
        class_id: {
            "id": class_id,
            "type": "Class",
            "name": class_name,
            "owner": None,
            "attributes": [attr_id],
            "methods": [method_id],
        },
        attr_id: {
            "id": attr_id,
            "type": "ClassAttribute",
            "name": f"+ {attr_name}: int",
            "owner": class_id,
        },
        method_id: {
            "id": method_id,
            "type": "ClassMethod",
            "name": f"+ {method_name}(qty: int): bool",
            "owner": class_id,
        },
    }

    constraint_elt: dict = {
        "id": constraint_id,
        "type": "ClassOCLConstraint",
        "name": "OCL",
        "owner": None,
        "constraint": constraint_text,
    }
    if legacy_kind is not None:
        constraint_elt["kind"] = legacy_kind
        if legacy_constraint_name is not None:
            constraint_elt["constraintName"] = legacy_constraint_name
        if legacy_kind in ("precondition", "postcondition"):
            constraint_elt["targetMethodId"] = method_id
    if description is not None:
        constraint_elt["description"] = description
    elements[constraint_id] = constraint_elt

    relationships: dict = {
        link_id: {
            "id": link_id,
            "type": "ClassOCLLink",
            "source": {"element": constraint_id},
            "target": {"element": class_id},
        }
    }

    if extra_classes:
        for cname, attrs in extra_classes:
            cid = _new_id()
            elements[cid] = {
                "id": cid,
                "type": "Class",
                "name": cname,
                "owner": None,
                "attributes": [],
                "methods": [],
            }
            for a in attrs.split(","):
                a = a.strip()
                if not a:
                    continue
                aid = _new_id()
                elements[cid]["attributes"].append(aid)
                elements[aid] = {
                    "id": aid,
                    "type": "ClassAttribute",
                    "name": a,
                    "owner": cid,
                }

    if duplicate_in_second_box:
        constraint2_id = _new_id()
        link2_id = _new_id()
        elements[constraint2_id] = {
            "id": constraint2_id,
            "type": "ClassOCLConstraint",
            "name": "OCL2",
            "owner": None,
            "constraint": constraint_text,
        }
        relationships[link2_id] = {
            "id": link2_id,
            "type": "ClassOCLLink",
            "source": {"element": constraint2_id},
            "target": {"element": class_id},
        }

    return {
        "title": "Stress",
        "model": {"elements": elements, "relationships": relationships},
    }


# ---------------------------------------------------------------------------
# Test case generators — target ~400 cases across the requested categories
# ---------------------------------------------------------------------------

def gen_cases() -> list[tuple[str, dict, str]]:
    """Returns list of (case_label, payload, category)."""
    cases: list[tuple[str, dict, str]] = []

    # 1) Malformed BOCL (canonical-shape constraints with broken bodies)
    malformed = [
        "context Book inv c1 self.pages > 0",                       # missing colon
        "Book inv c1: self.pages > 0",                              # missing 'context'
        "context Book inv c1: (self.pages > 0",                     # mismatched parens
        "context Book inv c1: self.pages > 0)",                     # extra close paren
        "context Book inv c1: self.pages >",                        # trailing operator
        "context Book inv c1:",                                     # empty body
        "context Book inv c1: >>>>",                                # only operators
        "context Book inv c1: lorem ipsum dolor sit amet",          # random text
        "context Book inv c1: self.pages and",                      # trailing 'and'
        "context Book inv c1: self.pages or or self.pages",         # double 'or'
        "context Book inv c1: self.pages > 0 and",                  # trailing and
        "context Book inv c1: self.pages > 0 implies",              # trailing implies
        "context Book inv c1: ((self.pages))",                      # double parens (should pass)
        "context Book inv c1: self.pages..0",                       # weird range
        "context Book inv c1: self.pages = =",                      # double equals
        "context Book inv c1: <= self.pages",                       # leading op
        "context Book inv c1: self.pages > 0 ;",                    # stray semicolon
        "context Book inv c1: self.pages > 'string'",               # mixed types
        "context Book inv c1: self->pages > 0",                     # wrong arrow
        "context Book inv c1: self..pages",                         # double dot
        "context Book inv: self.pages > 0",                         # nameless inv (legal? log either way)
        "context Book inv c1 self.pages",                           # missing colon + body broken
        "context Book inv c1: self.pages > 0 self.pages",           # two exprs
        "context Book inv c1: 1 + ",                                # arithmetic stub
        "context Book inv c1: not",                                 # bare not
        "context Book inv c1: forAll(",                             # half operator
        "context Book inv c1: collect(x |)",                        # half lambda
        "context Book inv c1: select()",                            # empty select
    ]
    for i, txt in enumerate(malformed):
        cases.append((f"malformed_{i:02d}", make_payload(txt), "malformed_BOCL"))

    # 2) Reference errors
    ref_err = [
        "context NoSuchClass inv c1: self.pages > 0",
        "context Book inv c1: self.nonexistent > 0",
        "context Book inv c1: self.pages.nope > 0",
        "context Book::nope() pre c1: self.pages > 0",
        "context Book::borrow() pre c1: nope > 0",
        "context Book::borrow(qty: int, extra: int) pre c1: qty > 0",  # wrong arity
        "context Book::borrow() post c1: result and self.pages",
        "context Book inv c1: self.PAGES > 0",                       # case mismatch
        "context book inv c1: self.pages > 0",                       # lowercase class
        "context Book inv: self.unknownAttr > 0",
        "context Book::missing(p: int) pre c1: p > 0",
        "context Book inv c1: self.pages.size() > 0",                # call on int
        "context Book inv c1: self.pages->collect(x | x.bad)",
        "context Book inv c1: self.pages->select(x | x = self.foo)",
        "context Book::borrow(qty: int) pre c1: qty.something > 0",
        "context Book::borrow() pre c1: self.pages > 0",            # missing param decl
    ]
    for i, txt in enumerate(ref_err):
        cases.append((f"refer_{i:02d}", make_payload(txt), "reference"))

    # 3) Grammar-rejected forms
    grammar = [
        "context Book pre c1: self.pages > 0",                      # pre without method
        "context Book post c1: self.pages > 0",                     # post without method
        "context Book init: self.pages = 0",                        # init unsupported
        "context Book def: foo() = 1",                              # def unsupported
        "context Book::borrow pre c1: qty > 0",                     # missing parens on method
        "context Book::borrow(qty: int) pre name1 name2: qty > 0",  # weird names
        "context Book::borrow() body: self.pages > 0",              # body keyword
        "context Book::borrow() pre: ",                             # empty pre
        "context Book inv inv: self.pages > 0",                    # repeated 'inv'
        "context context Book inv c1: self.pages > 0",              # double context
    ]
    for i, txt in enumerate(grammar):
        cases.append((f"grammar_{i:02d}", make_payload(txt), "grammar"))

    # 4) Whitespace / layout
    whitespace = [
        "    context Book inv c1: self.pages > 0",                  # leading spaces
        "context Book inv c1: self.pages > 0      ",                # trailing spaces
        "context Book\r\ninv c1: self.pages > 0",                   # CRLF inside
        "\r\n\r\ncontext Book inv c1: self.pages > 0\r\n",          # surrounding CRLF
        "\n\n\n",                                                    # only newlines
        "-- only a comment\n",                                      # only comments
        "context Book inv c1: true",                                # literal true
        "context Book inv c1: false",                               # literal false
        "\t\tcontext Book inv c1: self.pages > 0",                  # tabs
        " \t \r\n context Book inv c1: self.pages > 0 ",            # mixed whitespace
        "",                                                          # empty
        "   ",                                                       # only spaces
        "-- comment only\n-- another\n",                            # multiple comments only
        "-- pre comment\ncontext Book inv c1: self.pages > 0",      # comment + valid
        "context Book inv c1: -- inline\nself.pages > 0",          # inline comment
    ]
    for i, txt in enumerate(whitespace):
        cases.append((f"ws_{i:02d}", make_payload(txt), "whitespace"))

    # 5) Unicode
    unicode_cases = [
        ("context Book inv c1: self.pages > 0", "über short"),
        ("context Book inv c1: self.pages > 0", "中文 description"),
        ("context Book inv c1: self.pages > 0", "💥💥💥 emoji"),
        ("context Book inv c1: self.pages > 0", "em — dash"),
        ("context Book inv c1: self.pages > 0", "café résumé naïve"),
        ("context Book inv c1: self.pages > 0", "x" * 1024),
        ("context Book inv c1: self.pages > 0", "y" * 4096),
        ("context Book inv c1: self.pages > 0", "Ω≈ç√∫˜µ"),
        # Unicode in attribute reference (will fail to resolve)
        ("context Book inv c1: self.üpages > 0", None),
        ("context Book inv c1: self.中文 > 0", None),
        ("context Book inv c1: self.\\u200bpages > 0", None),  # zero-width
        ("context Bóók inv c1: self.pages > 0", None),
    ]
    for i, (txt, desc) in enumerate(unicode_cases):
        cases.append((f"unicode_{i:02d}", make_payload(txt, description=desc), "unicode"))

    # 6) Boundary sizes
    sizes = []
    sizes.append("a")                                                 # 1 char
    sizes.append("context Book inv c1: " + "(" * 15 + "self.pages > 0" + ")" * 15)  # deeply nested
    sizes.append("context Book inv c1: " + "(" * 25 + "self.pages > 0" + ")" * 25)
    chained = "context Book inv c1: self.pages" + "->collect(x | x)" * 50
    sizes.append(chained)
    chained2 = "context Book inv c1: self.pages" + ("->collect(x | x)->select(x | x > 0)" * 25)
    sizes.append(chained2)
    sizes.append("X" * 10240)                                        # 10KB random text
    sizes.append("context Book inv c1: " + "self.pages > 0 and " * 200 + "true")
    sizes.append("context Book inv c1: " + "1 + " * 500 + "1")
    sizes.append("(" * 200 + ")" * 200)                              # only parens
    sizes.append("context Book inv c1: " + "(" * 50 + "true" + ")" * 50)
    for i, txt in enumerate(sizes):
        cases.append((f"size_{i:02d}", make_payload(txt), "boundary"))

    # 7) Concurrency / safety — duplicate constraints in two boxes
    dup = [
        "context Book inv dup1: self.pages > 0",
        "context Book inv dup2: self.pages >",  # broken duplicated
        "context Book inv dup3: self.pages and",
    ]
    for i, txt in enumerate(dup):
        cases.append((
            f"dup_{i:02d}",
            make_payload(txt, duplicate_in_second_box=True),
            "duplicate",
        ))

    # 8) Legacy body-only shape
    legacy = [
        ("self.pages > 0", "invariant", "inv_legacy_1"),
        ("self.pages >", "invariant", "inv_legacy_broken"),
        ("self.pages > 0 and", "invariant", "inv_legacy_trailing"),
        ("qty > 0", "precondition", "pre_legacy_ok"),
        ("qty >", "precondition", "pre_legacy_broken"),
        ("result = true", "postcondition", "post_legacy_ok"),
        ("nonsense >>>>", "postcondition", "post_legacy_garbage"),
        ("", "invariant", "empty_body_legacy"),
        ("   ", "invariant", "ws_body_legacy"),
        ("self.pages > 0", "WRONG_KIND", "bad_kind"),
        ("self.pages > 0", "invariant", None),  # no name
        ("self.pages > 0", "precondition", "pre_no_method"),  # method targeting handled
    ]
    for i, (body, kind, name) in enumerate(legacy):
        cases.append((
            f"legacy_{i:02d}",
            make_payload(body, legacy_kind=kind, legacy_constraint_name=name),
            "legacy",
        ))

    # 9) Cross-class pre/post — body references another class's attr
    extra = [("Author", "+ name: str")]
    cross = [
        "context Book inv c1: self.author.name = 'x'",
        "context Book inv c1: Author.allInstances()->size() > 0",
        "context Book::borrow(qty: int) pre c1: Author.allInstances()->size() = qty",
        "context Book inv c1: self.author.bogus > 0",
        "context Book inv c1: NoSuchClass.allInstances()->size() > 0",
    ]
    for i, txt in enumerate(cross):
        cases.append((
            f"cross_{i:02d}",
            make_payload(txt, extra_classes=extra),
            "cross_class",
        ))

    # 10) Random-fuzz extras to push toward 400+
    fuzz = [
        "context", "context Book", "context Book inv",
        ":", "::", ":::", "context Book::",
        "context Book::borrow",
        "self.pages > 0",                              # body without header
        "inv c1: self.pages > 0",                     # missing context
        "context inv c1: self.pages > 0",             # missing class name
        "context Book inv c1: \\n\\n",
        "context Book inv c1: \"unterminated",
        "context Book inv c1: 'unterminated",
        "context Book inv c1: /* C-style comment */ self.pages > 0",
        "context Book inv c1: <html>",
        "context Book inv c1: <script>alert(1)</script>",
        "context Book inv c1: '; DROP TABLE Book; --",
        "context Book inv c1: ${jndi:ldap://x}",
        "context Book inv c1: \\x00\\x01",
        "context Book inv c1: \x00",                  # actual NUL
        "context Book inv c1: \x07",                  # bell
        "context Book inv c1:\nself.pages\n>\n0",     # multi-line broken
        "context Book inv c1: if self.pages then 1 else 0 endif",
        "context Book inv c1: let x = 0 in x > -1",
        "context Book inv c1: Set{1,2,3}->size() = 3",
        "context Book inv c1: Sequence{1,2}->first()",
        "context Book inv c1: Tuple{a=1, b=2}",
        "context Book inv c1: self.pages = self.pages",
        "context Book inv c1: self <> null",
        "context Book inv c1: self.oclIsKindOf(Book)",
        "context Book inv c1: self.oclIsTypeOf(Author)",
        "context Book inv c1: self.oclAsType(Book).pages > 0",
        "context Book inv c1: 1 / 0",
        "context Book inv c1: self.pages mod 0",
        "context Book inv c1: self.pages div 0",
    ]
    for i, txt in enumerate(fuzz):
        cases.append((f"fuzz_{i:02d}", make_payload(txt), "fuzz"))

    # Multiply some categories to reach ~400 by parametric variations
    base_bodies = [
        "context Book inv c{n}: self.pages > {n}",
        "context Book inv c{n}: self.pages > 0 and self.pages < {n}",
        "context Book::borrow(qty: int) pre c{n}: qty > {n}",
        "context Book::borrow(qty: int) post c{n}: result implies qty > {n}",
        "context Book inv c{n}: self.pages = {n}",
    ]
    for n in range(50):
        for i, tmpl in enumerate(base_bodies):
            cases.append((
                f"valid_{i}_{n:02d}",
                make_payload(tmpl.format(n=n)),
                "valid_baseline",
            ))

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> None:
    cases = gen_cases()
    total = len(cases)

    crashes: list[tuple[str, str, str]] = []         # (label, where, traceback)
    warned_processor: list[str] = []                  # produced warning during process_class_diagram
    invalid_at_validator: list[tuple[str, list[str]]] = []
    valid_at_validator: list[tuple[str, list[str]]] = []
    silent: list[str] = []                            # no warnings, no invalid -- check whether legitimate
    opaque_traceback_msgs: list[tuple[str, str]] = [] # invalid-constraints msg that looks like a traceback
    failure_modes: dict[str, tuple[str, str]] = {}    # category -> (input, snippet)

    t0 = time.time()
    for label, payload, category in cases:
        # Stage 1: process_class_diagram
        dm = None
        try:
            dm = process_class_diagram(payload)
        except Exception:
            tb = traceback.format_exc()
            crashes.append((label, "process_class_diagram", tb))
            failure_modes.setdefault(f"crash_{category}", (label, tb.splitlines()[-1]))
            continue

        warns = list(getattr(dm, "ocl_warnings", []) or [])
        if warns:
            warned_processor.append(label)

        # Stage 2: check_ocl_constraint
        try:
            res = check_ocl_constraint(dm)
        except Exception:
            tb = traceback.format_exc()
            crashes.append((label, "check_ocl_constraint", tb))
            failure_modes.setdefault(f"crash_validator_{category}", (label, tb.splitlines()[-1]))
            continue

        invalid = res.get("invalid_constraints") or []
        valid = res.get("valid_constraints") or []
        if invalid:
            invalid_at_validator.append((label, invalid))
            for msg in invalid:
                # Heuristic: opaque traceback string indicators
                if "Traceback" in msg or "  File \"" in msg or "0x" in msg:
                    opaque_traceback_msgs.append((label, msg[:200]))
        if valid:
            valid_at_validator.append((label, valid))
        if not warns and not invalid and not valid:
            silent.append(label)

        # Track first example per failure category
        if invalid:
            failure_modes.setdefault(f"invalid_{category}", (label, str(invalid[0])[:240]))
        elif warns:
            failure_modes.setdefault(f"warn_{category}", (label, str(warns[0])[:240]))

    elapsed = time.time() - t0

    # Report
    print("=" * 80)
    print("OCL PIPELINE STRESS TEST REPORT")
    print("=" * 80)
    print(f"Total cases run               : {total}")
    print(f"Pipeline crashes (uncaught)   : {len(crashes)}")
    print(f"Cases warned by processor     : {len(warned_processor)}")
    print(f"Cases invalid at validator    : {len(invalid_at_validator)}")
    print(f"Cases valid at validator      : {len(valid_at_validator)}")
    print(f"Cases silent (no warn/invalid): {len(silent)}")
    print(f"Opaque-traceback messages     : {len(opaque_traceback_msgs)}")
    print(f"Wall time                     : {elapsed:.2f}s")
    print()

    if crashes:
        print("CRASHES:")
        for lbl, where, tb in crashes[:20]:
            print(f"  - [{lbl}] {where}")
            print(tb[-400:])
        print()
    else:
        print("NO CRASHES. Pipeline did not raise an uncaught exception on any input.")
        print()

    print("Sample failure modes (one per category):")
    for k, (label, snippet) in list(failure_modes.items())[:25]:
        snippet_oneline = snippet.replace("\n", " ")[:200]
        print(f"  - {k:40s} | {label:20s} | {snippet_oneline}")

    if opaque_traceback_msgs:
        print()
        print("Opaque-traceback samples:")
        for lbl, msg in opaque_traceback_msgs[:5]:
            print(f"  - [{lbl}] {msg}")

    if silent:
        print()
        print(f"Silent-pass examples (first 10 of {len(silent)}):")
        for lbl in silent[:10]:
            print(f"  - {lbl}")

    # Fail loud if any crash
    if crashes:
        print()
        print(">>> AUDIT FAILED: pipeline crashed on at least one input. <<<")
        sys.exit(2)


if __name__ == "__main__":
    run()
