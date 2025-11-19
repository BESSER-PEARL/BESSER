import os
import textwrap

from jinja2 import Environment, FileSystemLoader
import json
import re

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.structural import Method
from besser.generators import GeneratorInterface


class BAFGenerator(GeneratorInterface):
    """
    BAFGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the agent code, using the BESSER Agent Framework (BAF), based on an input agent model.

    Args:
        model (Agent): A agent model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: Agent, output_dir: str = None):
        super().__init__(model, output_dir)

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

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        env.globals['is_class'] = is_class
        env.globals['is_type'] = is_type
        env.globals['replace_bot_session_with_session_in_signature'] = replace_agent_session_with_session_in_signature
        agent_template = env.get_template('baf_agent_template.py.j2')
        agent_path = self.build_generation_path(file_name=f"{self.model.name}.py")
        with open(agent_path, mode="w", encoding="utf-8") as f:
            # todo: how to handle llm variable names that are used in bodies?
            generated_code = agent_template.render(agent=self.model)
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

