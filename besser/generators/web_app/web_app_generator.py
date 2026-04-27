import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.gui import GUIModel
from besser.BUML.metamodel.structural import DomainModel
from besser.generators.backend import BackendGenerator
from besser.generators.react import ReactGenerator
from besser.generators import GeneratorInterface
from besser.utilities.buml_code_builder.common import safe_var_name


def agent_slug(name) -> str:
    """Canonical filesystem slug for an agent.

    Reused across the web-app generator (``agents/<slug>/`` directories,
    docker-compose build contexts) and the GitHub/Render deploy pipeline.
    Accepts either a string or an object exposing ``.name``.

    Filesystem paths, container names, and hostnames are conventionally
    lowercase. ``lowercase=True`` is currently ``safe_var_name``'s default,
    but we pass it explicitly here to document the requirement and to keep
    this slug stable if the default ever changes.
    """
    raw = getattr(name, "name", name) if not isinstance(name, str) else name
    return safe_var_name(raw, lowercase=True) if raw else "agent"


##############################
#   Web Application Generator
##############################
class WebAppGenerator(GeneratorInterface):
    """
    WebAppGenerator is responsible for generating a web application structure based on
    input B-UML and GUI models. It implements the GeneratorInterface and facilitates
    the creation of a web application.

    Args:
        model (DomainModel): The B-UML model representing the application's domain.
        gui_model (GUIModel): The GUI model instance containing necessary configurations.
        output_dir (str, optional): Directory where generated code will be saved. Defaults to None.
        agent_models (list, optional): List of agent models. Each agent is generated into
            ``agents/<name>/`` under the output directory.
        agent_configs (dict, optional): Mapping of ``agent.name`` -> config dict, passed
            per agent to the underlying :class:`BAFGenerator`.
        agent_model: **Deprecated.** Single-agent back-compat shim. Prefer ``agent_models``.
        agent_config: **Deprecated.** Single-agent back-compat shim. Prefer ``agent_configs``.
    """

    def __init__(self, model: DomainModel, gui_model: GUIModel, output_dir: str = None,
                 agent_models=None, agent_configs=None,
                 agent_model=None, agent_config=None):
        super().__init__(model, output_dir)
        self.gui_model = gui_model
        if agent_model is not None and not agent_models:
            agent_models = [agent_model]
        if agent_config is not None and not agent_configs:
            name = getattr(agent_model, "name", "agent") if agent_model is not None else "agent"
            agent_configs = {name: agent_config}
        self.agent_models = list(agent_models or [])
        self.agent_configs = dict(agent_configs or {})
        # Jinja environment configuration
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True,
                               lstrip_blocks=True, extensions=['jinja2.ext.do'])
        self.env.filters['agent_slug'] = agent_slug

    @property
    def agent_model(self):
        """Deprecated: returns the first agent model, if any."""
        return self.agent_models[0] if self.agent_models else None

    @property
    def agent_config(self):
        """Deprecated: returns the config of the first agent, if any."""
        if not self.agent_models:
            return None
        return self.agent_configs.get(getattr(self.agent_models[0], "name", None))

    def generate(self):
        """
        Generates web application code based on the provided B-UML and GUI models.
        If any agent models are provided, also generates agent code (one per agent).

        Returns:
            None, but store the generated code in the specified output directory.
        """
        self._generate_frontend(self.env)
        self._generate_backend(self.env)
        if self.agent_models:
            self._generate_agent(self.env)
        self._generate_docker_files(self.env)

    def _generate_frontend(self, env):
        # Generate frontend code in 'frontend' subfolder
        frontend_dir = os.path.join(self.output_dir, "frontend") if self.output_dir else "frontend"
        frontend_gen = ReactGenerator(self.model, self.gui_model, output_dir=frontend_dir)
        frontend_gen.generate()

    def _generate_backend(self, env):
        # Generate backend code in 'backend' subfolder
        backend_dir = os.path.join(self.output_dir, "backend") if self.output_dir else "backend"
        backend_gen = BackendGenerator(self.model, output_dir=backend_dir)
        backend_gen.generate()

    def _generate_agent(self, env):
        """Generate agent code for every configured agent model.

        Each agent is emitted into ``agents/<slug>/`` under the output directory
        so that multi-agent projects don't collide.
        """
        from besser.generators.agents.baf_generator import BAFGenerator

        agents_root = os.path.join(self.output_dir, "agents") if self.output_dir else "agents"
        os.makedirs(agents_root, exist_ok=True)

        seen_slugs = set()
        for agent in self.agent_models:
            slug = agent_slug(agent)
            # Defensive dedupe in case the upstream uniqueness guards fail.
            base = slug
            suffix = 2
            while slug in seen_slugs:
                slug = f"{base}_{suffix}"
                suffix += 1
            seen_slugs.add(slug)

            agent_dir = os.path.join(agents_root, slug)
            os.makedirs(agent_dir, exist_ok=True)
            cfg = self.agent_configs.get(getattr(agent, "name", None))
            BAFGenerator(agent, output_dir=agent_dir, config=cfg).generate()

    def _generate_docker_files(self, env):
        """
        Generates Docker-related files for deployment.

        Args:
            env: The Jinja2 environment for template rendering.
        """
        # Generate docker-compose.yml
        docker_compose_template = env.get_template('docker-compose.yml.j2')
        docker_compose_path = os.path.join(self.output_dir, 'docker-compose.yml')
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_template.render(
                agent_models=self.agent_models,
                # Back-compat: keep the scalar variable populated so any out-of-tree
                # template fork that still reads ``agent_model`` keeps working.
                agent_model=self.agent_models[0] if self.agent_models else None,
                name=self.model.name,
            ))

        # Generate frontend Dockerfile
        frontend_dockerfile_template = env.get_template('frontend.Dockerfile.j2')
        frontend_dockerfile_path = os.path.join(self.output_dir, 'frontend', 'Dockerfile')
        os.makedirs(os.path.dirname(frontend_dockerfile_path), exist_ok=True)
        with open(frontend_dockerfile_path, 'w') as f:
            f.write(frontend_dockerfile_template.render())

        # Generate backend Dockerfile
        backend_dockerfile_template = env.get_template('backend.Dockerfile.j2')
        backend_dockerfile_path = os.path.join(self.output_dir, 'backend', 'Dockerfile')
        os.makedirs(os.path.dirname(backend_dockerfile_path), exist_ok=True)
        with open(backend_dockerfile_path, 'w') as f:
            f.write(backend_dockerfile_template.render())

        # Generate one Dockerfile per agent under agents/<slug>/Dockerfile
        if self.agent_models:
            agent_dockerfile_template = env.get_template('agent.Dockerfile.j2')
            for agent in self.agent_models:
                slug = agent_slug(agent)
                agent_dockerfile_path = os.path.join(self.output_dir, 'agents', slug, 'Dockerfile')
                os.makedirs(os.path.dirname(agent_dockerfile_path), exist_ok=True)
                with open(agent_dockerfile_path, 'w') as f:
                    f.write(agent_dockerfile_template.render(agent_model=agent))
