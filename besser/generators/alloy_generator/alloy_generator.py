"""AlloyGenerator: translates BESSER/BUML domain models to Alloy specifications."""

import os
import re
from collections import defaultdict

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    DomainModel,
    Enumeration,
)
from besser.generators import GeneratorInterface
from besser.generators.alloy_generator.translate_ocl_alloy import (
    EstadoTraductor,
    generate_dates_and_order,
    ocl_to_alloy,
)

ALLOY_IDENTIFIER_REGEX = re.compile(r"[^A-Za-z0-9_]")

ALLOY_KEYWORDS = {
    "abstract", "all", "and", "as", "assert", "but", "check", "disj", 
    "else", "enum", "exactly", "expt", "fact", "for", "fun", "iden", 
    "iff", "implies", "in", "int", "Int", "let", "lone", "module", 
    "no", "none", "not", "one", "open", "or", "pred", "run", "seq", 
    "set", "sig", "some", "sum", "univ"
}

def sanitize_alloy_name(name: str) -> str:
    """Return a valid Alloy identifier derived from *name*.
    Args:
        name: Raw identifier to sanitize.
    Returns:
        A non-empty string that is a legal Alloy identifier.
    """
    sanitized = ALLOY_IDENTIFIER_REGEX.sub("", name)
    if not sanitized:
        return "_unnamed"
    if sanitized[0].isdigit() or sanitized in ALLOY_KEYWORDS:
        sanitized = "_" + sanitized
    return sanitized


def build_consistency_rule(
    clase_a: str,
    rel_a_b: str,
    mult_b: list,
    clase_b: str,
    rel_b_a: str,
    mult_a: list,
    flecha_a_b: bool,
    flecha_b_a: bool,
) -> str:
    """Build the Alloy cardinality-consistency facts for one association end.

    Emits ``fact`` blocks when the multiplicity differs from the implicit
    ``1..1`` default.  When the navigation direction of an end is enabled, its
    facts navigate the field directly (``a.<A>_<rel>``); when the direction is
    not navigable but the opposite end is, the facts navigate the opposite
    field in reverse (``<B>_<rel>.a``) so the multiplicity still holds.  If
    neither end is navigable there is no field to express the relation and no
    fact is emitted.

    Args:
        clase_a:   Name of class A (source side).
        rel_a_b:   Role name navigating from A to B.
        mult_b:    ``[min, max]`` multiplicity for the B side.
        clase_b:   Name of class B (target side).
        rel_b_a:   Role name navigating from B to A.
        mult_a:    ``[min, max]`` multiplicity for the A side.
        flecha_a_b: ``True`` when A→B is navigable.
        flecha_b_a: ``True`` when B→A is navigable.

    Returns:
        A string containing zero or more Alloy ``fact`` declarations.
    """
    res = "\n"
    MULTIPLICITY_LIMIT = 9999  # Arbitrary upper limit for multiplicity; Alloy uses "unbounded" for "*".
    # B-side multiplicity: how many B instances each A is related to.
    if not (mult_b[0] == 1 and mult_b[1] == 1):
        if flecha_a_b:
            nav = f"a.{clase_a}_{rel_a_b}"
        elif flecha_b_a:
            nav = f"{clase_b}_{rel_b_a}.a"
        else:
            nav = None
        if nav:
            if mult_b[0] >= 1 and mult_b[0] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all a: {clase_a} | #({nav})>={mult_b[0]} }}"
            if mult_b[1] >= 1 and mult_b[1] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all a: {clase_a} | #({nav})<={mult_b[1]} }}"

    # A-side multiplicity: how many A instances each B is related to.
    if not (mult_a[0] == 1 and mult_a[1] == 1):
        if flecha_b_a:
            nav = f"b.{clase_b}_{rel_b_a}"
        elif flecha_a_b:
            nav = f"{clase_a}_{rel_a_b}.b"
        else:
            nav = None
        if nav:
            if mult_a[0] >= 1 and mult_a[0] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all b: {clase_b} | #({nav})>={mult_a[0]} }}"
            if mult_a[1] >= 1 and mult_a[1] < MULTIPLICITY_LIMIT:
                res += f"\nfact{{all b: {clase_b} | #({nav})<={mult_a[1]} }}"

    return res


class AlloyGenerator(GeneratorInterface):
    """Generate an Alloy specification file from a BESSER/BUML ``DomainModel``.

    The generator renders Jinja2 templates to produce a ``.als`` file containing:

    * Basic type signatures (``str``, ``Int``, enumerations).
    * One Alloy ``sig`` per class, with attributes and navigable association ends.
    * Cardinality consistency ``fact`` blocks for non-default multiplicities.
    * Inverse-relation ``fact`` blocks for bidirectional associations.
    * OCL constraints translated to Alloy ``fact`` blocks.
    * A generic ``pred instance_model`` with a ``run`` command.
    """

    def __init__(self, model: DomainModel, output_dir: str | None = None, scope: int = 5):
        super().__init__(model, output_dir)
        self.scope = scope

    def generate(self) -> None:
        """Render all templates and write the ``.als`` file to *output_dir*."""
        file_path = self.build_generation_path(file_name="model.als")

        spec: list[str] = []
        data: dict = defaultdict(list)
        inherits_from: dict = defaultdict(list)
        basic_signatures: set = set()

        #In sigs_nv is a list of all class and basic type names later used into template alloy_final_als.j2 to declare the scope of each signature
        sigs_nv: list[str] = []

        # Sanitize class and attribute names in-place
        for class_obj in self.model.classes_sorted_by_inheritance():
            class_obj.name = sanitize_alloy_name(class_obj.name)
            sigs_nv.append(class_obj.name)
            for attr in class_obj.attributes:
                attr.name = sanitize_alloy_name(attr.name)

        # Build inheritance and attribute maps
        for class_obj in self.model.classes_sorted_by_inheritance():
            if len(class_obj.parents()) == 0:
                inherits_from[class_obj.name].append("_")
            else:
                for parent in class_obj.parents():
                    inherits_from[class_obj.name].append(parent.name)

            for attr in class_obj.attributes:
                tipo = "date" if attr.type.name in ("date", "datetime", "time", "timedelta") else attr.type.name
                data[class_obj.name].append(f"{attr.name}:{tipo}")
                if not isinstance(attr.type, Enumeration):
                    basic_signatures.add(tipo)
                    sigs_nv.append(tipo)

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do"],
        )

        template = env.get_template("alloy_signatures_basic.j2")
        facts_rules: list[str] = []

        for assoc in self.model.associations:
            d, h = assoc.ends
            mult_b = [h.multiplicity.min, h.multiplicity.max]
            mult_a = [d.multiplicity.min, d.multiplicity.max]
            flecha_a_b = bool(h.is_navigable)
            flecha_b_a = bool(d.is_navigable)

            facts_rules.append(
                build_consistency_rule(
                    d.type.name, h.name, mult_b,
                    h.type.name, d.name, mult_a,
                    flecha_a_b, flecha_b_a,
                )
            )
            data[h.type.name].append(f"{d.name}:{d.type.name}")
            data[d.type.name].append(f"{h.name}:{h.type.name}")

            if flecha_a_b and flecha_b_a:
                facts_rules.append(
                    f"fact{{{d.type.name}_{h.name}= ~{h.type.name}_{d.name}}}"
                )

        # enums is a dictionary mapping enumeration names to sets of their literal names.
        enums = {
            enum_obj.name: {lit.name for lit in (enum_obj.literals or set())}
            for enum_obj in self.model.elements
            if isinstance(enum_obj, Enumeration)
        }

        # State stores necessary information preserved across calls.
        estado = EstadoTraductor()
        for constraint in self.model.constraints:
            context = constraint.context.name
            ocl_str = constraint.expression.split(":", 1)[1]
            constraint.expression = ocl_to_alloy(inherits_from, data, ocl_str, context, estado, enums)

        # Complete the date universe up to the scope and pin the total order
        # whenever the model has date-typed attributes or OCL date literals.
        date_block = ""
        if estado.fechas or "date" in basic_signatures:
            date_block = generate_dates_and_order(estado.fechas, self.scope)

        
        initial_signatures = template.render(
            basic_signatures=basic_signatures,
            enum_types={x for x in self.model.elements if isinstance(x, Enumeration)},
            has_date_values=bool(estado.fechas) or ("date" in basic_signatures),
        )
        spec.append(initial_signatures)
        if date_block:
            spec.append(date_block)

        for class_obj in self.model.classes_sorted_by_inheritance():
            template = env.get_template("alloy_model.j2")
            relevant_associations = [
                assoc for assoc in self.model.associations
                if any(end.type.name == class_obj.name for end in assoc.ends)
            ]
            generated_code = template.render(class_obj=class_obj, associations=relevant_associations)
            spec.append(generated_code)

        template = env.get_template("alloy_final_als.j2")

        fun_types: list = []

        final = template.render(
            fun_types=fun_types,
            constraints=self.model.constraints,
            sigsnv=sigs_nv,
            scope=self.scope,   
        )
        spec.append(final)

        with open(file_path, "w") as archivo:
            archivo.writelines(spec)
            facts_r = "\n" + "\n".join(facts_rules)
            archivo.writelines(facts_r)
