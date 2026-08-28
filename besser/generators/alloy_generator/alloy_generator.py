
import os
from collections import defaultdict

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    DomainModel,
    Enumeration,
)
from besser.generators import GeneratorInterface
from besser.generators.alloy_generator.translate_ocl_alloy import (
    TranslatorState,
    ocl_to_alloy,
)
from besser.generators.alloy_generator.utils_alloy import (
    build_consistency_rule,
    collect_enumerations,
    generate_date_block,
    sanitize_alloy_name,
)

class AlloyGenerator(GeneratorInterface):
    """
    AlloyGenerator: translates BESSER/BUML domain models to Alloy specifications.

    Current implementation translates class diagrams into Alloy models.

    The generator renders Jinja2 templates to produce a ``.als`` file containing:
    - Type signatures for basic or standard built-in datatypes (``str``, ``Int``, enumerations).
    - Signatures that represent classes, with fields that represent class attributes and
        navigable association ends.
    - Facts that enforce cardinality constraints for non-default multiplicities.
    - Facts enforcing enforcing transpose relational equivalence for bidirectional associations.
    - Facts capturing OCL constraints in the model.
    - A predicate without any additional constraints, to be used for model consistency checking.
    - A run command associated with the above predicate.
    """

    def __init__(self, model: DomainModel, output_dir: str | None = None, scope: int = 5):
        """
            Constructor for AlloyGenerator. Takes the domain model, output directory, and scope as parameters.
        """
        super().__init__(model, output_dir)
        self.scope = scope
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sanitize_model_names(self) -> None:
        """
        Sanitizes class and attribute names in-place for Alloy compatibility.
        """
        for class_obj in self.model.classes_sorted_by_inheritance():
            class_obj.name = sanitize_alloy_name(class_obj.name)
            for attr in class_obj.attributes:
                attr.name = sanitize_alloy_name(attr.name)

    def _build_inheritance_and_attribute_maps(self):
        """
        Builds inheritance, attribute, and signature maps.

        Returns:
            A tuple ``(inherits_from, data, basic_signatures, sigs_nv)``.
        """
        inherits_from: dict = defaultdict(list)
        data: dict = defaultdict(list)
        basic_signatures: set = set()
        sigs_nv: list[str] = []

        for class_obj in self.model.classes_sorted_by_inheritance():
            sigs_nv.append(class_obj.name)

            if len(class_obj.parents()) == 0:
                inherits_from[class_obj.name].append("_")
            else:
                for parent in class_obj.parents():
                    inherits_from[class_obj.name].append(parent.name)

            for attr in class_obj.attributes:
                attr_type = "date" if attr.type.name in ("date", "datetime", "time", "timedelta") else attr.type.name
                data[class_obj.name].append(f"{attr.name}:{attr_type}")
                if not isinstance(attr.type, Enumeration):
                    basic_signatures.add(attr_type)
                    sigs_nv.append(attr_type)

        return inherits_from, data, basic_signatures, sigs_nv

    def _process_associations(self, data: dict) -> list[str]:
        """
        Processes associations, building consistency facts and updating *data*.

        Returns:
            A list of Alloy fact strings for associations.
        """
        facts_rules: list[str] = []

        for assoc in self.model.associations:
            d, h = assoc.ends
            mult_b = [h.multiplicity.min, h.multiplicity.max]
            mult_a = [d.multiplicity.min, d.multiplicity.max]
            arrow_a_b = bool(h.is_navigable)
            arrow_b_a = bool(d.is_navigable)

            facts_rules.append(
                build_consistency_rule(
                    d.type.name, h.name, mult_b,
                    h.type.name, d.name, mult_a,
                    arrow_a_b, arrow_b_a,
                )
            )
            data[h.type.name].append(f"{d.name}:{d.type.name}")
            data[d.type.name].append(f"{h.name}:{h.type.name}")

            if arrow_a_b and arrow_b_a:
                facts_rules.append(
                    f"fact{{{d.type.name}_{h.name}= ~{h.type.name}_{d.name}}}"
                )

        return facts_rules

    def _translate_constraints(
        self, inherits_from: dict, data: dict, enums: dict
    ) -> TranslatorState:
        """
        Translates OCL constraints to Alloy facts in-place.

        Returns:
            A :class:`TranslatorState` object, carrying accumulated state (e.g. date
            literals discovered during translation).
        """
        state = TranslatorState()
        for constraint in self.model.constraints:
            context = constraint.context.name
            ocl_str = constraint.expression.split(":", 1)[1]
            constraint.expression = ocl_to_alloy(
                inherits_from, data, ocl_str, context, state, enums
            )
        return state

    def _render_spec(
        self,
        basic_signatures: set,
        sigs_nv: list[str],
        enums: dict,
        estado: TranslatorState,
        date_block: str,
        facts_rules: list[str],
    ) -> str:
        """
        Renders all Jinja2 templates and appends association facts.

        Returns:
            The complete ``.als`` specification as a single string.
        """
        parts: list[str] = []

        template = self.env.get_template("alloy_signatures_basic.j2")
        initial_signatures = template.render(
            basic_signatures=basic_signatures,
            enum_types={x for x in self.model.elements if isinstance(x, Enumeration)},
            has_date_values=bool(estado.dates) or ("date" in basic_signatures),
        )
        parts.append(initial_signatures)
        if date_block:
            parts.append(date_block)

        for class_obj in self.model.classes_sorted_by_inheritance():
            template = self.env.get_template("alloy_model.j2")
            relevant_associations = [
                assoc for assoc in self.model.associations
                if any(end.type.name == class_obj.name for end in assoc.ends)
            ]
            generated_code = template.render(class_obj=class_obj, associations=relevant_associations)
            parts.append(generated_code)

        template = self.env.get_template("alloy_final_als.j2")
        final = template.render(
            fun_types=[],
            constraints=self.model.constraints,
            sigsnv=sigs_nv,
            scope=self.scope,
        )
        parts.append(final)

        if facts_rules:
            parts.append("\n" + "\n".join(facts_rules))

        return "".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> None:
        """
        Renders all templates and writes the ``.als`` file to *output_dir*.
        """
        file_path = self.build_generation_path(file_name="model.als")

        self._sanitize_model_names()
        inherits_from, data, basic_signatures, sigs_nv = (
            self._build_inheritance_and_attribute_maps()
        )
        facts_rules = self._process_associations(data)
        enums = collect_enumerations(self.model)
        estado = self._translate_constraints(inherits_from, data, enums)
        date_block = generate_date_block(estado, basic_signatures, self.scope)
        spec = self._render_spec(
            basic_signatures, sigs_nv, enums, estado, date_block, facts_rules
        )

        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(spec)
