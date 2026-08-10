"""Generator that emits a pytest + Hypothesis test suite from a B-UML DomainModel."""

import os
import re
from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp
from besser.utilities.buml_code_builder.common import _escape_python_string


def _regex_findall(value, pattern):
    return re.findall(pattern, value)


def _regex_replace(value, find, replace):
    return re.sub(find, replace, value)


class TestCaseGenerator(GeneratorInterface):
    """
    TestCaseGenerator implements GeneratorInterface and generates a
    pytest + Hypothesis test suite from a B-UML DomainModel.

    Produces:
      - Section 1: Structural plain-pytest tests (class shape, attributes,
        properties, methods, constructor signatures, basic instantiation,
        setter round-trips, name constraint enforcement).
      - Section 2: Hypothesis property-based tests (st.builds strategies,
        instantiation invariants, type contracts, method @given tests,
        association set tests).
      - Section 3: OCL post-condition tests translated from each method's
        post-conditions.

    Args:
        model (DomainModel): The B-UML domain model to generate tests from.
        output_dir (str, optional): Output directory. Defaults to None.
        module_name (str, optional): Module the generated tests import the
            domain classes from (``from <module_name> import ...``). Matches
            the ``PythonGenerator`` default output module. Defaults to
            ``"classes"``.
    """

    # Stop pytest from trying to collect this class as a test suite (its name
    # starts with "Test"); it is a generator, not a test case.
    __test__ = False

    def __init__(self, model: DomainModel, output_dir: str = None, module_name: str = "classes"):
        super().__init__(model, output_dir)
        self.module_name = module_name

    def generate(self) -> None:
        """Generate ``test_hypothesis.py`` in the configured output directory."""
        file_path = self.build_generation_path(file_name="test_hypothesis.py")
        templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
        env = Environment(loader=FileSystemLoader(templates_path))
        # Custom filters used by hypothesis_tests_template.py.j2:
        # - to_strategy: maps B-UML primitive type names -> Hypothesis strategies
        # - escape_py: escapes model-controlled text embedded in string literals
        # - regex_findall / regex_replace: used by the OCL constraint section
        env.filters["to_strategy"] = buml_type_to_hypothesis_strategy
        env.filters["escape_py"] = _escape_python_string
        env.filters["regex_findall"] = _regex_findall
        env.filters["regex_replace"] = _regex_replace
        template = env.get_template("hypothesis_tests_template.py.j2")
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                domain=self.model,
                module_name=self.module_name,
                sort_by_timestamp=sort_by_timestamp,
            )
            f.write(generated_code)


def buml_type_to_hypothesis_strategy(type_name: str) -> str:
    """
    Maps a B-UML primitive type name to the matching Hypothesis strategy string.
    Constrained to avoid nan/inf for floats. Falls back to st.none() for
    unknown/complex types so the template stays valid.
    """
    mapping = {
        "str": "safe_text",
        "int": "st.integers()",
        "float": "st.floats(allow_nan=False, allow_infinity=False)",
        "bool": "st.booleans()",
        "date": "st.dates()",
        "datetime": "st.datetimes()",
        "time": "st.times()",
        "timedelta": "st.timedeltas()",
    }
    return mapping.get(type_name, "st.none()")
