"""Docker Compose generator — turns a UML DeploymentModel into docker-compose.yml."""
import os
import re

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Locality,
)
from besser.generators import GeneratorInterface
from besser.generators.agents.baf_generator import BAFGenerator
from besser.utilities import sort_by_timestamp


def _safe_service_name(name: str) -> str:
    """Convert a deployment element label to a Docker Compose-safe service/network name.

    Applies camelCase → snake_case conversion first, then lowercases everything
    and replaces runs of non-alphanumeric characters with a single underscore.

    Examples: "AgentRuntime" → "agent_runtime", "llm_gateway" → "llm_gateway",
              "LLMEndpoint" → "llm_endpoint", "Code Tester" → "code_tester".
    """
    s = (name or "").strip()
    # Insert underscore between a lowercase/digit and an uppercase (e.g. tR → t_R)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Insert underscore between a run of uppercase and the start of a new word
    # e.g. "LLMEndpoint" → "LLM_Endpoint"
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s or 'unnamed'


class DockerComposeGenerator(GeneratorInterface):
    """Generate a docker-compose.yml from a UML DeploymentModel.

    Maps UML Deployment elements to Compose services and networks following
    the §3 table in the 05-docker-compose-generator-guide:

    - Artifact → service (locality decides ``build:`` vs ``image:``)
    - DeploymentRelation.multiplicity.max → ``deploy.replicas`` (when > 1)
    - Node → named network under ``networks:``
    - CommunicationPath → bridging ``<a>_<b>_link`` network both nodes join
    - DeploymentDependency (Artifact → Artifact) → ``depends_on:``
    - Interface / InterfaceProvided / InterfaceRequired → not mapped (v1)

    ``Artifact.manifests`` is emitted as a ``# manifests:`` comment only (v1
    — not resolved against the Component model; see guide §9 Q3).
    """

    def __init__(self, model: DeploymentModel, output_dir: str = None,
                 agent_models_by_id: dict = None):
        super().__init__(model, output_dir)
        # 6b-2 — {AgentDiagram-uuid → BUML Agent model}, supplied by the
        # project-level router handler. Empty on the single-diagram path, in
        # which case no build contexts are baked (6a compose-only behavior).
        self.agent_models_by_id = agent_models_by_id or {}

    def generate(self):
        file_path = self.build_generation_path(file_name="docker-compose.yml")
        templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        services, networks = self._build_view(self.model)
        template = env.get_template("docker-compose.yml.j2")
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(template.render(services=services, networks=networks))
        print("Code generated in the location: " + file_path)
        # 6b-2 — bake a BAF build context per resolvable agentic LOCAL artifact.
        self._bake_agent_contexts(env)

    def _bake_agent_contexts(self, env: Environment) -> None:
        """For each LOCAL Artifact carrying a resolvable ``agent_model_ref``,
        bake a build context (``<output_dir>/<svc_name>/``) containing the BAF
        ``agent.py`` + ``config.yaml`` (via BAFGenerator) and a ``Dockerfile``.

        The directory name is ``_safe_service_name(art.name)`` — identical to the
        ``build: ./<svc_name>`` the compose emits for this artifact, so the two
        line up. No-ops when ``agent_models_by_id`` is empty (single-diagram
        path). LOCAL artifacts with no/unresolvable ref are skipped (logged).
        """
        if not self.agent_models_by_id:
            return
        base_dir = os.path.dirname(
            self.build_generation_path(file_name="docker-compose.yml")
        )
        dockerfile_tpl = env.get_template("Dockerfile.j2")
        for art in self.model.all_artifacts():
            if art.locality != Locality.LOCAL:
                continue
            ref = getattr(art, "agent_model_ref", None)
            if not ref:
                continue
            agent = self.agent_models_by_id.get(ref)
            if agent is None:
                print(f"[docker_compose] artifact '{art.name}' references agent "
                      f"'{ref}' but no matching AgentDiagram was found — "
                      f"skipping its build context.")
                continue
            svc_name = _safe_service_name(art.name)
            ctx_dir = os.path.join(base_dir, svc_name)
            os.makedirs(ctx_dir, exist_ok=True)
            # BAF agent.py + config.yaml into the build context.
            BAFGenerator(agent, output_dir=ctx_dir).generate()
            # Dockerfile referencing the agent script BAFGenerator just wrote.
            agent_script = f"{agent.name}.py"
            with open(os.path.join(ctx_dir, "Dockerfile"),
                      mode="w", encoding="utf-8") as f:
                f.write(dockerfile_tpl.render(agent_script=agent_script))
            print(f"[docker_compose] baked build context: {ctx_dir}")

    def _build_view(self, model: DeploymentModel) -> tuple:
        """Resolve the metamodel into ordered dicts the template renders.

        Doing the graph walk here keeps the template declarative and lets tests
        assert on the view dicts directly.  Returns ``(services, networks)`` —
        plain-dict lists, deterministic order via ``sort_by_timestamp``.
        """
        all_artifacts = list(sort_by_timestamp(model.all_artifacts()))
        all_nodes = list(sort_by_timestamp(model.all_nodes()))
        all_rels = list(sort_by_timestamp(model.relationships))

        node_to_net = {id(n): _safe_service_name(n.name) for n in all_nodes}

        # ----- Pass 1: artifact → set of node python-ids it's deployed on ---
        # Sources: explicit DeploymentRelations AND containment (parent).
        art_nodes_map: dict = {id(a): set() for a in all_artifacts}

        for rel in all_rels:
            if isinstance(rel, DeploymentRelation):
                src_key = id(rel.source)
                if src_key in art_nodes_map:
                    art_nodes_map[src_key].add(id(rel.target))

        for art in all_artifacts:
            if art.parent is not None:
                art_nodes_map[id(art)].add(id(art.parent))

        # ----- Pass 2: networks per artifact (from node membership) ----------
        art_nets: dict = {id(a): [] for a in all_artifacts}

        def _add_net(art_key: int, net: str) -> None:
            if net not in art_nets[art_key]:
                art_nets[art_key].append(net)

        for art in all_artifacts:
            for node_id in art_nodes_map[id(art)]:
                net = node_to_net.get(node_id)
                if net:
                    _add_net(id(art), net)

        # ----- Pass 3: replicas from DeploymentRelation.multiplicity ---------
        art_replicas: dict = {}
        for rel in all_rels:
            if isinstance(rel, DeploymentRelation):
                src_key = id(rel.source)
                if src_key not in art_replicas:
                    mult = rel.multiplicity
                    if mult.max > 1:
                        art_replicas[src_key] = mult.max

        # ----- Pass 4: CommunicationPath → bridging networks ----------------
        cp_nets: list = []
        seen_cp: set = set()
        for rel in all_rels:
            if isinstance(rel, CommunicationPath):
                src_id = id(rel.source)
                tgt_id = id(rel.target)
                src_net = node_to_net.get(src_id, _safe_service_name(rel.source.name))
                tgt_net = node_to_net.get(tgt_id, _safe_service_name(rel.target.name))
                link = f"{src_net}_{tgt_net}_link"
                if link not in seen_cp:
                    seen_cp.add(link)
                    cp_nets.append(link)
                for art in all_artifacts:
                    if (src_id in art_nodes_map[id(art)]
                            or tgt_id in art_nodes_map[id(art)]):
                        _add_net(id(art), link)

        # ----- Pass 5: depends_on from DeploymentDependency -----------------
        art_safe = {id(a): _safe_service_name(a.name) for a in all_artifacts}
        art_depends: dict = {id(a): [] for a in all_artifacts}
        for rel in all_rels:
            if isinstance(rel, DeploymentDependency):
                if not (isinstance(rel.source, Artifact)
                        and isinstance(rel.target, Artifact)):
                    continue
                src_key = id(rel.source)
                if src_key in art_depends:
                    tgt_svc = art_safe.get(id(rel.target),
                                           _safe_service_name(rel.target.name))
                    if tgt_svc not in art_depends[src_key]:
                        art_depends[src_key].append(tgt_svc)

        # ----- Build service dicts -------------------------------------------
        services = []
        for art in all_artifacts:
            svc_name = _safe_service_name(art.name)
            art_key = id(art)

            if art.locality == Locality.LOCAL:
                build = f"./{svc_name}"
                image = None
                is_hybrid = False
            elif art.locality == Locality.EXTERNAL:
                build = None
                image = f"{svc_name}:latest"
                is_hybrid = False
            else:  # HYBRID
                build = None
                image = f"{svc_name}:latest"
                is_hybrid = True

            services.append({
                'name': svc_name,
                'build': build,
                'image': image,
                'is_hybrid': is_hybrid,
                'networks': art_nets.get(art_key, []),
                'depends_on': art_depends.get(art_key, []),
                'replicas': art_replicas.get(art_key),
                'stereotypes': ', '.join(art.stereotypes) if art.stereotypes else None,
                'manifests': ', '.join(art.manifests) if art.manifests else None,
            })

        # ----- Build network dicts ------------------------------------------
        seen_nets: set = set()
        networks = []
        for node in all_nodes:
            net_name = node_to_net[id(node)]
            if net_name not in seen_nets:
                seen_nets.add(net_name)
                kind_comment = f"  # kind: {node.kind.value}" if node.kind else ""
                networks.append({'name': net_name, 'kind_comment': kind_comment})
        for link in cp_nets:
            if link not in seen_nets:
                seen_nets.add(link)
                networks.append({'name': link, 'kind_comment': ""})

        return services, networks
