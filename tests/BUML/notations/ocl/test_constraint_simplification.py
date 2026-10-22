"""Chapter 3 simplification, and the comment that keeps the original text.

CorrectProduct' is the worked example in section 3.3 of Cabot and Teniente,
*Incremental Integrity Checking in UML/OCL Conceptual Schemas*. The
normalized body is the CorrectProduct invariant from Figure 1.2.
"""

import pytest

from besser.BUML.notations.ocl import parse_ocl, simplify_constraint
from tests.BUML.notations.ocl.test_incremental_preanalysis import (
    ecommerce_model,
)


CORRECT_PRODUCT_ORIGINAL = (
    "context Product inv CorrectProduct: "
    "not Product.allInstances()->exists(p | p.price <= 0 or p.maxDiscount > 60)"
)
CORRECT_PRODUCT_NORMAL = (
    "context Product inv CorrectProduct: self.price > 0 and self.maxDiscount <= 60"
)
ALREADY_SIMPLE = (
    "context Category inv AtLeastThreeCustomers: self.customer->size() >= 3"
)


def test_correct_product_prime_becomes_the_figure_1_2_form():
    model = ecommerce_model()
    constraint = parse_ocl(CORRECT_PRODUCT_ORIGINAL, model)
    constraint.name = "CorrectProduct"
    constraint.expression = CORRECT_PRODUCT_ORIGINAL

    original = simplify_constraint(constraint, model)

    assert original == CORRECT_PRODUCT_ORIGINAL
    assert constraint.expression == CORRECT_PRODUCT_NORMAL
    assert constraint.metadata.description == CORRECT_PRODUCT_ORIGINAL
    assert constraint.description is None


def test_an_unchanged_expression_gets_no_comment():
    model = ecommerce_model()
    constraint = parse_ocl(ALREADY_SIMPLE, model)
    constraint.name = "AtLeastThreeCustomers"
    constraint.expression = ALREADY_SIMPLE

    assert simplify_constraint(constraint, model) is None
    assert constraint.expression == ALREADY_SIMPLE
    assert constraint.metadata is None


def test_spacing_alone_does_not_count_as_a_change():
    model = ecommerce_model()
    compact = "context Product inv CorrectProduct: self.price>0 and self.maxDiscount<=60"
    constraint = parse_ocl(compact, model)
    constraint.expression = compact

    assert simplify_constraint(constraint, model) is None
    assert constraint.expression == compact
    assert constraint.metadata is None


def test_simplifying_again_does_not_add_a_second_comment():
    model = ecommerce_model()
    constraint = parse_ocl(CORRECT_PRODUCT_ORIGINAL, model)
    constraint.expression = CORRECT_PRODUCT_ORIGINAL
    simplify_constraint(constraint, model)

    assert simplify_constraint(constraint, model) is None
    assert constraint.metadata.description == CORRECT_PRODUCT_ORIGINAL


def _diagram(texts: dict[str, str]) -> dict:
    from besser.utilities.web_modeling_editor.backend.services.converters import class_buml_to_json

    model = ecommerce_model()
    for name, text in texts.items():
        constraint = parse_ocl(text, model)
        constraint.name = name
        constraint.expression = text
        model.add_constraint(constraint)
    return {"title": "Ecommerce", "model": class_buml_to_json(model)}


def test_the_editor_payload_reports_only_the_constraints_that_changed():
    from besser.utilities.web_modeling_editor.backend.services.ocl.constraint_simplification import (
        constraint_simplification,
    )

    payload = _diagram({
        "CorrectProduct": CORRECT_PRODUCT_ORIGINAL,
        "AtLeastThreeCustomers": ALREADY_SIMPLE,
    })
    result = constraint_simplification(payload)
    by_text = {item.get("original"): item for item in result["constraints"] if item["changed"]}

    assert result["warnings"] == []
    assert len(by_text) == 1
    changed = by_text[CORRECT_PRODUCT_ORIGINAL]
    assert changed["expression"] == CORRECT_PRODUCT_NORMAL
    assert changed["elementId"]
    unchanged = [item for item in result["constraints"] if not item["changed"]]
    assert len(unchanged) == 1
    assert "expression" not in unchanged[0]
    assert "original" not in unchanged[0]


def test_a_linked_comment_round_trips_with_the_normalized_expression():
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        class_buml_to_json,
        process_class_diagram,
    )
    from besser.utilities.web_modeling_editor.backend.services.ocl.constraint_simplification import (
        constraint_simplification,
    )

    payload = _diagram({"CorrectProduct": CORRECT_PRODUCT_ORIGINAL})
    result = constraint_simplification(payload)
    changed = next(item for item in result["constraints"] if item["changed"])
    box = payload["model"]["elements"][changed["elementId"]]
    box["constraint"] = changed["expression"]
    comment_id = "original-expression"
    payload["model"]["elements"][comment_id] = {
        "id": comment_id,
        "name": changed["original"],
        "type": "Comments",
        "owner": None,
        "bounds": {"x": 0, "y": 0, "width": 160, "height": 80},
    }
    payload["model"]["relationships"]["link-original"] = {
        "id": "link-original",
        "name": "",
        "type": "Link",
        "owner": None,
        "source": {"direction": "Left", "element": comment_id, "multiplicity": "", "role": ""},
        "target": {"direction": "Right", "element": changed["elementId"], "multiplicity": "", "role": ""},
    }

    loaded = process_class_diagram(payload)
    constraint = next(c for c in loaded.constraints if c.name == "CorrectProduct")
    assert constraint.expression == CORRECT_PRODUCT_NORMAL
    assert constraint.metadata.description == CORRECT_PRODUCT_ORIGINAL
    assert constraint.description is None

    exported = class_buml_to_json(loaded)
    comments = [element for element in exported["elements"].values() if element.get("type") == "Comments"]
    assert [element["name"] for element in comments] == [CORRECT_PRODUCT_ORIGINAL]
    links = [
        relationship for relationship in exported["relationships"].values()
        if relationship.get("type") == "Link"
    ]
    assert len(links) == 1
    ends = {links[0]["source"]["element"], links[0]["target"]["element"]}
    boxes = [
        element for element in exported["elements"].values()
        if element.get("type") == "ClassOCLConstraint"
    ]
    assert len(boxes) == 1
    assert boxes[0]["constraint"] == CORRECT_PRODUCT_NORMAL
    assert "description" not in boxes[0]
    assert comments[0]["id"] in ends
    assert boxes[0]["id"] in ends


def test_python_export_keeps_the_original_expression_on_the_constraint(tmp_path):
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code

    model = ecommerce_model()
    constraint = parse_ocl(CORRECT_PRODUCT_ORIGINAL, model)
    constraint.name = "CorrectProduct"
    constraint.expression = CORRECT_PRODUCT_ORIGINAL
    model.add_constraint(constraint)
    simplify_constraint(constraint, model)
    path = tmp_path / "ecommerce.py"
    domain_model_to_code(model, str(path))
    namespace: dict = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    restored = next(c for c in namespace["domain_model"].constraints if c.name == "CorrectProduct")
    assert restored.expression == CORRECT_PRODUCT_NORMAL
    assert restored.metadata.description == CORRECT_PRODUCT_ORIGINAL


def test_simplification_rejects_a_diagram_that_is_not_a_class_diagram():
    from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError
    from besser.utilities.web_modeling_editor.backend.services.ocl.constraint_simplification import (
        constraint_simplification,
    )

    with pytest.raises(ConversionError):
        constraint_simplification({
            "title": "Flow",
            "model": {"type": "StateMachineDiagram", "elements": {}},
        })
