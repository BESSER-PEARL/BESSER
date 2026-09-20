
import copy
import os

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    DomainModel,
    Enumeration,
)
from besser.generators import GeneratorInterface
from besser.generators.alloy.alloy_utils_generator import (
    build_inheritance_and_attribute_maps,
    generate_date_block,
    generate_utils_module,
    process_associations,
    sanitize_model_names,
    translate_constraints,
)
from besser.generators.alloy.string_ops import build_string_sigs
from besser.generators.alloy.translate_ocl_alloy import (
    DATES_DICT,
    resolve_ocl_date_literals,
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
    - Facts enforcing transpose relational equivalence for bidirectional associations.
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
        self.template = self.env.get_template("alloy_spec.j2")

    def generate(self) -> None:
        """
        Generates an Alloy specification based on the provided B-UML model and saves it to
        the specified output directory.
        If the output directory was not specified, the code generated will be stored in the
        <current directory>/output folder.

        Stores the generated specification in a file named model.als
        """
        file_path = self.build_generation_path(file_name="model.als")

        model = copy.deepcopy(self.model)

        sanitize_model_names(model)

        inherits_from, data, basic_signatures, sigs_nv = (
            build_inheritance_and_attribute_maps(model)
        )
        facts_rules = process_associations(model, data)

        enum_types = {el for el in model.elements if isinstance(el, Enumeration)}
        enums = {e.name: {lit.name for lit in (e.literals or set())} for e in enum_types}

        status = translate_constraints(model, inherits_from, data, enums)
        date_block = generate_date_block(status, basic_signatures, self.scope)
        if status.dates and DATES_DICT:
            resolve_ocl_date_literals(model.constraints)

        has_string_sigs = bool(status.strings)
        has_str_types = any(t in {"str", "string", "Str"} for t in basic_signatures)
        needs_str_ops = has_str_types
        string_block = build_string_sigs(status.strings) if has_string_sigs else ""
        needs_date_ops = bool(status.dates) or ("date" in basic_signatures)
        classes = model.classes_sorted_by_inheritance()
        associations_by_class = {c.name: [] for c in classes}
        for assoc in model.associations:
            for end in assoc.ends:
                if end.type.name in associations_by_class:
                    if assoc not in associations_by_class[end.type.name]:
                        associations_by_class[end.type.name].append(assoc)

        maxseq = max(self.scope, status.maxseq)
        use_str = has_str_types
        int_bitwidth = max(self.scope, maxseq.bit_length() + 1) if use_str else self.scope

        spec = self.template.render(
            basic_signatures=basic_signatures,
            enum_types=enum_types,
            has_date_values=bool(status.dates) or ("date" in basic_signatures),
            classes=classes,
            associations_by_class=associations_by_class,
            constraints=model.constraints,
            sigsnv=sigs_nv,
            scope=self.scope,
            maxseq=maxseq,
            int_bitwidth=int_bitwidth,
            facts_rules=facts_rules,
            string_ops=needs_str_ops,
            date_ops=needs_date_ops,
        )

        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(spec)
        generate_utils_module(os.path.dirname(file_path))

        if needs_str_ops:
            status.string_ops.generate_str_ops_model(
                os.path.dirname(file_path), string_block
            )

        if needs_date_ops:
            status.date_ops.generate_date_ops_model(
                os.path.dirname(file_path), date_block
            )
