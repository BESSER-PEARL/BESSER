"""Phase 2 must not weaken security code to make a check pass.

Production run 438889bc: a bcrypt/passlib mismatch made registration fail,
and the model rewrote password hashing trying to satisfy the check instead of
finding the dependency that caused it.
"""
from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt


def test_prompt_forbids_weakening_security_to_pass_a_check():
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    prompt = build_system_prompt(
        DomainModel(name="M", types={user}), None, None, inventory="",
        instructions="Add login", max_turns=10,
    )
    flat = " ".join(prompt.split())
    assert ("Never weaken authentication, password hashing, crypto, authorisation "
            "or input validation to make a check or test pass") in flat
    assert "find the cause (dependency, configuration, test input) and fix that" in flat
