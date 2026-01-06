import os
import textwrap

from jinja2 import Environment, FileSystemLoader
import json
import re

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.structural import Method
from besser.generators import GeneratorInterface

from besser.generators.agents.agent_personalization import personalize_agent, configure_agent

# BESSER utilities
from besser.utilities.buml_code_builder import (
    agent_model_to_code,
)
from besser.utilities.web_modeling_editor.backend.services.converters import agent_buml_to_json

class BAFGenerator(GeneratorInterface):
    """
    BAFGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the agent code, using the BESSER Agent Framework (BAF), based on an input agent model.

    Args:
        model (Agent): A agent model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: Agent, output_dir: str = None, config_path: str = None, config: dict = None):
        super().__init__(model, output_dir)
        self.config = config
        if config_path:
            print("Loading config from:", config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

    def generate(self):
        """
        Generates the BAF agent code and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.
        """

        # TODO: TelegramPlatform.add_handler() not implemented in generator
        # TODO: Verify imports are added when necessary
        # TODO: Platform name not safe (hardcoded 'websocket_platform' and 'telegram_platform' in jinja template), when accessed from body can be different name
        # TODO: Global variables?
        # -->   (OPTION 1) agent.create_global_var(name: str, type: type, value: Any) --> not supports "custom" values (e.g. x = agent.get_name() )
        # -->   (OPTION 2) agent.add_code_line('x = agent.get_name()')

        def is_class(obj, name):
            return obj.__class__.__name__ == name

        def is_type(obj, type_name: str):
            return type(obj).__name__ == type_name

        def replace_agent_session_with_session_in_signature(func: Method) -> str:
            if func:
                # Replace 'AgentSession' with 'Session' in the code
                code = func.code.replace('AgentSession', 'Session')
                # Extract function name using regex
                match = re.search(r'def\s+(\w+)\s*\(.*session\s*:\s*Session.*\)', code)
                if match:
                    code = code + "\n" + f"{func.name} = {match.group(1)}\n"
                return textwrap.dedent(code)
            else:
                return None

        def rag_slug(name: str, index: int) -> str:
            slug = (name or '').strip().lower().replace(' ', '_').replace('-', '_')
            if not slug:
                slug = f"rag_{index}"
            return slug

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        env.globals['is_class'] = is_class
        env.globals['is_type'] = is_type
        env.globals['replace_bot_session_with_session_in_signature'] = replace_agent_session_with_session_in_signature
        agent_template = env.get_template('baf_agent_template.py.j2')
        agent_path = self.build_generation_path(file_name=f"{self.model.name}.py")
        personalized_agent_path = self.build_generation_path(file_name="personalized_agent_model.py")
        personalized_json_path = self.build_generation_path(file_name="personalized_agent_model.json")
        personalized_messages = {}
        print(self.config)
        if self.config:
            if 'personalizationrules' in self.config:
                personalize_agent(self.model, self.config['personalizationrules'], personalized_messages)
            else:
                configure_agent(self.model, self.config)

            # Persist personalized agent python for downstream conversion
            agent_model_to_code(self.model, personalized_agent_path)

            # Also emit JSON representation of personalized agent
            try:
                with open(personalized_agent_path, "r", encoding="utf-8") as f:
                    personalized_code = f.read()
                personalized_json = agent_buml_to_json(personalized_code)
                with open(personalized_json_path, "w", encoding="utf-8") as jf:
                    json.dump(personalized_json, jf, indent=2)
                print("Personalized agent JSON generated in the location: " + personalized_json_path)
            except Exception as conversion_error:
                print(f"Failed to convert personalized agent to JSON: {conversion_error}")

        if personalized_messages == {}:
            
            with open(agent_path, mode="w", encoding="utf-8") as f:
                # todo: how to handle llm variable names that are used in bodies?
                generated_code = agent_template.render(agent=self.model, config=self.config)
                f.write(generated_code)
                print("Agent script generated in the location: " + agent_path)
        else: 
            with open(agent_path, mode="w", encoding="utf-8") as f:
                print(self.config['personalizationrules'])
                print(personalized_messages)
                generated_code = agent_template.render(agent=self.model, config=self.config, personalized_messages=personalized_messages)
                f.write(generated_code)
                print("Agent script generated in the location: " + agent_path)
        config_template = env.get_template('baf_config_template.py.j2')
        config_path = self.build_generation_path(file_name="config.ini")
        with open(config_path, mode="w", encoding="utf-8") as f:
            properties = sorted(self.model.properties, key=lambda prop: prop.section)
            generated_code = config_template.render(properties=properties)
            f.write(generated_code)
            print("Agent config file generated in the location: " + config_path)        # Generate readme.txt using the Jinja2 template
        readme_template = env.get_template('readme.txt.j2')
        readme_path = self.build_generation_path(file_name="readme.txt")
        with open(readme_path, mode="w", encoding="utf-8") as f:
            generated_code = readme_template.render(agent=self.model)
            f.write(generated_code)
            print("Agent readme file generated in the location: " + readme_path)

        rag_configs = getattr(self.model, 'rags', []) or []
        if rag_configs:
            rag_base_dir = self.build_generation_dir()
            for idx, rag in enumerate(rag_configs):
                target_dir = os.path.join(rag_base_dir, rag_slug(getattr(rag, 'name', ''), idx))
                os.makedirs(target_dir, exist_ok=True)
                readme_path = os.path.join(target_dir, "README.txt")
                if not os.path.exists(readme_path):
                    with open(readme_path, "w", encoding="utf-8") as readme_file:
                        readme_file.write("Place your PDF documents for this RAG database inside this folder before running the agent.\n")
