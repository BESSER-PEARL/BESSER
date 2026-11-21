"""The Phase 2 rules describe the run they are shown to.

Rules 4-6 (state machines, object instances, OCL constraints) were sent to
every run, whatever models it had; most runs have none of them. They are now
emitted only for the models the run has, with the rule numbers unchanged so
references to a rule still hold. Rule 6's cross-record part stays: rules the
planner reads from the request need it without an OCL model. The colour map is
shortened, the CRUD rule lets a request that names its screens win, and the
planner's examples no longer come from one application's request.
"""

from types import SimpleNamespace

from besser.BUML.metamodel.structural import Class, Constraint, DomainModel, Method, PrimitiveDataType, Property
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.planning.gap_analyzer import _build_user_prompt
from besser.spec_driven_agent.planning.target import WEB_APP, Target

WEB = Target(WEB_APP, "web app", True)


def _model(constraint=False):
    order = Class(name="Order")
    order.attributes = {Property(name="total", type=PrimitiveDataType("float"))}
    constraints = {Constraint(name="positiveTotal", context=order, language="OCL",
                              expression="context Order inv positiveTotal: self.total > 0")} if constraint else set()
    return DomainModel(name="Shop", types={order}, constraints=constraints)


def _prompt(**kwargs):
    model = kwargs.pop("domain_model", _model())
    return build_system_prompt(model, kwargs.pop("gui_model", None), None, "", "Build a shop web app", 10,
                               target=WEB, **kwargs)


def test_a_run_without_those_models_gets_no_model_rules():
    prompt = _prompt()
    assert "4. **State machines" not in prompt
    assert "5. **Object instances" not in prompt
    assert "OCL constraints must run" not in prompt
    # Rule 6's cross-record part still applies to rules read from the request.
    assert "6. **Rules must run.**" in prompt and "shared transactional enforcement" in prompt
    assert prompt.index("3. **The user's original") < prompt.index("6. **") < prompt.index("7. **")


def test_each_model_brings_its_rule_under_its_own_number():
    assert "6. **OCL constraints must run.**" in _prompt(domain_model=_model(constraint=True))
    prompt = _prompt(state_machines=[SimpleNamespace(name="OrderLifecycle", states=[], transitions=[])])
    assert "## State Machines" in prompt and "4. **State machines drive code.**" in prompt


def test_a_method_contract_alone_brings_the_ocl_rule():
    # The converter attaches pre/postconditions to the method, never to model.constraints.
    model = _model()
    order, = model.get_classes()
    pay = Method(name="pay")
    pay.add_pre(Constraint(name="payable", context=order, language="OCL",
                           expression="context Order::pay() pre: self.total > 0"))
    order.methods = {pay}
    assert not model.constraints
    assert "6. **OCL constraints must run.**" in _prompt(domain_model=model)


def test_the_colour_rule_keeps_the_failure_it_guards_and_drops_the_long_map():
    for prompt in (_prompt(), _prompt(design_system="## Design system\n")):
        assert "`color: rose` renders nothing" in prompt and "#f43f5e" in prompt
        assert "#ec4899" not in prompt


def test_a_request_that_names_its_screens_wins_over_per_entity_crud():
    # The checklist item's own wording: the request scopes the UI, not a screen named in passing.
    assert "or the request scopes the UI to the screens it\n    names: then build those." in _prompt()
    gui = SimpleNamespace(modules=[SimpleNamespace(
        name="Main", screens=[SimpleNamespace(name="Home", view_elements=set())])])
    hinted = _prompt(gui_model=gui)
    assert "LOOSE HINT" in hinted and "Screens the request names come first" in hinted


def test_the_planner_examples_name_no_single_application():
    prompt = _build_user_prompt("req", "generate_fastapi_backend", "{}", "inv").lower()
    for word in ("booking", "bill", "room"):
        assert word not in prompt
