"""A regenerable generator's output is a pure function of the model.

regenerate() checks the stored output by generating its model again, in another
process; any difference turns the three-way merge off for the app.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Method, MethodImplementationType,
    Multiplicity, Property, StringType,
)
from besser.spec_driven_agent.pipeline.constants import REGENERABLE_GENERATORS

ROOT = pathlib.Path(__file__).resolve().parents[2]

LOOP_METHOD = """def large_orders(threshold:int) -> int {
    count = 0;
    for(order in this.orders){
        for(other in this.orders){
            if(order.total > threshold){
                count = count + 1;
            }
        }
    }
    return count;
}"""


def loop_model() -> DomainModel:
    """Customer with a BAL method holding nested for loops over its Orders."""
    order = Class(name="Order", attributes={Property(name="total", type=IntegerType)})
    customer = Class(name="Customer", attributes={Property(name="name", type=StringType)})
    method = Method(name="large_orders", parameters=[], type=None,
                    implementation_type=MethodImplementationType.BAL)
    method.code = LOOP_METHOD
    customer.add_method(method)
    places = BinaryAssociation(name="Places", ends={
        Property(name="orders", type=order, multiplicity=Multiplicity(0, "*")),
        Property(name="customer", type=customer, multiplicity=Multiplicity(0, 1))})
    return DomainModel(name="Shop", types={order, customer}, associations={places})


def digests(generator: str) -> dict[str, str]:
    from besser.spec_driven_agent.pipeline.constants import _APPLICATION_OUTPUT_DIRS
    from besser.spec_driven_agent.pipeline.regeneration import _generate

    files, error = _generate(generator, loop_model(), _APPLICATION_OUTPUT_DIRS[generator])
    assert files, error
    return {path: hashlib.sha256(content).hexdigest() for path, content in sorted(files.items())}


def _digests_in_new_process(generator: str) -> dict[str, str]:
    script = ("import json, sys\n"
              "from tests.spec_driven_agent.test_generator_determinism import digests\n"
              f"print(json.dumps(digests({generator!r})))\n")
    run = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def _differing(first: dict[str, str], second: dict[str, str]) -> list[str]:
    return sorted(path for path in first.keys() | second.keys() if first.get(path) != second.get(path))


@pytest.mark.parametrize("generator", sorted(REGENERABLE_GENERATORS))
def test_generator_output_is_the_same_in_every_process(generator):
    """BAL for-loop variables were named after an object hash, so
    routers/<class>_methods.py differed between two generations."""
    here = digests(generator)

    assert not _differing(here, digests(generator))
    assert not _differing(here, _digests_in_new_process(generator))
    assert not _differing(here, _digests_in_new_process(generator))
