from besser.generators.alloy.alloy_translator_new import BumlToAlloy
from tests.generators.alloy.utils_alloy_re_parser import AlloySpec
from besser.BUML.metamodel.structural.structural import Constraint

def translate(model, classname: str, constraint: str, scope: int) -> AlloySpec:
    model.constraints = {
        Constraint(
            name="named",
            context=model.get_class_by_name(classname),
            expression=f"context {classname} inv named: {constraint}",
            language="OCL",
        )
    }
    spec = BumlToAlloy(model, scope=scope).translate()
    return AlloySpec(spec.render())