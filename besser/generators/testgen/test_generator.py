import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp


class TestGenerator(GeneratorInterface):
    """
    HypothesisTestGenerator implements GeneratorInterface and generates a
    pytest + Hypothesis test suite from a B-UML DomainModel.

    Produces:
      - Section 1: Structural plain-pytest tests (class shape, attributes,
        properties, methods, constructor signatures, basic instantiation,
        setter round-trips, name constraint enforcement).
      - Section 2: Hypothesis property-based tests (st.builds strategies,
        instantiation invariants, type contracts, method @given tests,
        association set tests).

    Args:
        model (DomainModel): The B-UML domain model to generate tests from.
        output_dir (str, optional): Output directory. Defaults to None.
    """

    def __init__(self, model: DomainModel, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self, *args):
        """
        Generates test_hypothesis.py in the output directory.
        """
        file_path = self.build_generation_path(file_name="test_hypothesis.py")
        templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
        env = Environment(loader=FileSystemLoader(templates_path))

        # Custom filter: maps B-UML primitive type names → Hypothesis strategies
        env.filters["to_strategy"] = buml_type_to_hypothesis_strategy

        template = env.get_template("hypothesis_tests_template.py.j2")
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                domain=self.model,
                sort_by_timestamp=sort_by_timestamp,
            )
            f.write(generated_code)
            print("Hypothesis tests generated in: " + file_path)


def buml_type_to_hypothesis_strategy(type_name: str) -> str:
    """
    Maps a B-UML primitive type name to the matching Hypothesis strategy string.
    Constrained to avoid nan/inf for floats. Falls back to st.none() for
    unknown/complex types so the template stays valid.
    """
    mapping = {
        "str":       "safe_text",
        "int":       "st.integers()",
        "float":     "st.floats(allow_nan=False, allow_infinity=False)",
        "bool":      "st.booleans()",
        "date":      "st.dates()",
        "datetime":  "st.datetimes()",
        "time":      "st.times()",
        "timedelta": "st.timedeltas()",
    }
    return mapping.get(type_name, "st.none()")