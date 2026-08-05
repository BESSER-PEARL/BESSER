import os
import pytest
from besser.BUML.metamodel.state_machine.agent import (
    Agent, Intent, LLMOpenAI, RAGReply, RAGTextSplitter, RAGVectorStore, WebSocketPlatform
)
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode


@pytest.fixture
def agent_model():
    """Create a minimal agent model with two states and a transition."""
    agent = Agent("test_agent")

    # Add a platform
    ws_platform = WebSocketPlatform()
    agent.platforms.append(ws_platform)

    # Define intents
    greet_intent = Intent(
        name="greet",
        training_sentences=["hello", "hi", "hey"],
    )
    agent.intents.append(greet_intent)

    # Create states using agent.new_state() so they are registered
    initial_state = agent.new_state(name="initial", initial=True)
    greeting_state = agent.new_state(name="greeting")

    # Add transition: initial -> greeting on greet intent
    initial_state.when_intent_matched(greet_intent).go_to(greeting_state)

    return agent


def test_baf_generator_instantiation(agent_model):
    """Test that the BAFGenerator can be instantiated."""
    generator = BAFGenerator(model=agent_model)
    assert generator is not None
    assert generator.generation_mode == GenerationMode.FULL


def test_baf_generator_code_only_mode(agent_model):
    """Test that CODE_ONLY generation mode is accepted."""
    generator = BAFGenerator(
        model=agent_model,
        generation_mode=GenerationMode.CODE_ONLY,
    )
    assert generator.generation_mode == GenerationMode.CODE_ONLY


def test_baf_generator_string_mode(agent_model):
    """Test that string generation mode is normalized correctly."""
    generator = BAFGenerator(
        model=agent_model,
        generation_mode="code_only",
    )
    assert generator.generation_mode == GenerationMode.CODE_ONLY


def test_baf_generator_generate(agent_model, tmpdir):
    """Test that generate() runs without errors and produces output."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    # The generator should create a Python file named after the agent
    agent_file = os.path.join(str(output_dir), f"{agent_model.name}.py")
    assert os.path.isfile(agent_file), f"{agent_model.name}.py should be generated"
    assert os.path.getsize(agent_file) > 0, "Generated agent file should not be empty"


def test_baf_generator_creates_config(agent_model, tmpdir):
    """Test that generate() creates a config.yaml file."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    config_file = os.path.join(str(output_dir), "config.yaml")
    assert os.path.isfile(config_file), "config.yaml should be generated"


def test_baf_generator_creates_readme(agent_model, tmpdir):
    """Test that generate() creates a readme.txt file."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    readme_file = os.path.join(str(output_dir), "readme.txt")
    assert os.path.isfile(readme_file), "readme.txt should be generated"


def test_baf_generator_output_content(agent_model, tmpdir):
    """Test that generated agent code contains expected content."""
    output_dir = tmpdir.mkdir("output")
    generator = BAFGenerator(
        model=agent_model,
        output_dir=str(output_dir),
        generation_mode=GenerationMode.CODE_ONLY,
    )
    generator.generate()

    agent_file = os.path.join(str(output_dir), f"{agent_model.name}.py")
    with open(agent_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "test_agent" in code, "Generated code should reference the agent name"


def _make_rag(name: str) -> tuple[RAGVectorStore, RAGTextSplitter]:
    vector_store = RAGVectorStore(
        embedding_provider="openai",
        embedding_parameters={"model": "text-embedding-3-small"},
        persist_directory=f"vector_store/{name}",
    )
    splitter = RAGTextSplitter(splitter_type="recursive_character", chunk_size=1000, chunk_overlap=100)
    return vector_store, splitter


@pytest.fixture
def multi_rag_agent_model():
    """Create an agent with two RAG configs and distinct RAGReply targets."""
    agent = Agent("multi_rag_agent")
    agent.platforms.append(WebSocketPlatform())
    LLMOpenAI(agent=agent, name="gpt-4o-mini", parameters={})

    products_vs, products_splitter = _make_rag("products")
    faq_vs, faq_splitter = _make_rag("faq")
    agent.new_rag(
        name="products_rag",
        vector_store=products_vs,
        splitter=products_splitter,
        llm_name="gpt-4o-mini",
        llm_prompt="Use only product catalog entries.",
    )
    agent.new_rag(name="faq_rag", vector_store=faq_vs, splitter=faq_splitter, llm_name="gpt-4o-mini")

    products_state = agent.new_state(name="products", initial=True)
    faq_state = agent.new_state(name="faq")

    products_state.set_body(Body("products_body", actions=[RAGReply("products_rag", prompt="Products only")]))
    faq_state.set_fallback_body(Body("faq_fallback", actions=[RAGReply("faq_rag")]))
    return agent


def test_baf_generator_uses_targeted_rag_instances(multi_rag_agent_model, tmp_path):
    """RAGReply emits ``<rag_var>.run(...)`` for body and fallback-body actions."""
    BAFGenerator(
        model=multi_rag_agent_model,
        output_dir=str(tmp_path),
        generation_mode=GenerationMode.CODE_ONLY,
    ).generate()

    agent_file = os.path.join(str(tmp_path), f"{multi_rag_agent_model.name}.py")
    with open(agent_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "rag_message = products_rag.run(session.event.message, llm_prompt='Products only')" in code
    assert "rag_message = faq_rag.run(session.event.message)" in code
    assert "rag_message = session.run_rag(session.event.message, llm_prompt='Products only')" not in code
    assert "llm_prompt='Use only product catalog entries.'" in code


def test_baf_generator_falls_back_to_session_rag_with_warning(tmp_path, caplog):
    """Unknown RAG names keep backward compatibility via ``session.run_rag``."""
    agent = Agent("rag_fallback_agent")
    agent.platforms.append(WebSocketPlatform())
    LLMOpenAI(agent=agent, name="gpt-4o-mini", parameters={})

    known_vs, known_splitter = _make_rag("known")
    agent.new_rag(name="known_rag", vector_store=known_vs, splitter=known_splitter, llm_name="gpt-4o-mini")

    fallback_state = agent.new_state(name="fallback", initial=True)
    fallback_state.set_body(Body("fallback_body", actions=[RAGReply("missing_rag", prompt="Use fallback")]))

    with caplog.at_level("WARNING"):
        BAFGenerator(
            model=agent,
            output_dir=str(tmp_path),
            generation_mode=GenerationMode.CODE_ONLY,
        ).generate()

    agent_file = os.path.join(str(tmp_path), f"{agent.name}.py")
    with open(agent_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "rag_message = session.run_rag(session.event.message, llm_prompt='Use fallback')" in code
    assert "references unknown rag_db_name 'missing_rag'" in caplog.text


def test_baf_generator_single_rag_still_uses_explicit_instance(tmp_path):
    """When ``rag_db_name`` resolves unambiguously, always emit ``rag.run(...)``."""
    agent = Agent("single_rag_agent")
    agent.platforms.append(WebSocketPlatform())
    LLMOpenAI(agent=agent, name="gpt-4o-mini", parameters={})

    only_vs, only_splitter = _make_rag("only")
    agent.new_rag(name="only_rag", vector_store=only_vs, splitter=only_splitter, llm_name="gpt-4o-mini")

    only_state = agent.new_state(name="only", initial=True)
    only_state.set_body(Body("only_body", actions=[RAGReply("only_rag")]))

    BAFGenerator(
        model=agent,
        output_dir=str(tmp_path),
        generation_mode=GenerationMode.CODE_ONLY,
    ).generate()

    agent_file = os.path.join(str(tmp_path), f"{agent.name}.py")
    with open(agent_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "rag_message = only_rag.run(session.event.message)" in code
    assert "rag_message = session.run_rag(session.event.message)" not in code


