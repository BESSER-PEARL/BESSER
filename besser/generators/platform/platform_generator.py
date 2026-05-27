"""
Platform Generator for BESSER

Generates a complete instance editor platform that allows users to create
instances of classes defined in a B-UML class diagram. The generated platform
includes a FastAPI backend and React + TypeScript frontend similar to the
BESSER web modeling editor.
"""

import ast
import os
import shutil
import textwrap
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.metamodel.platform_customization import PlatformCustomizationModel
from besser.generators import GeneratorInterface
from besser.generators.platform.template_helpers import (
    build_representation_registries,
    build_subclass_registry,
    compute_addable_port_classes,
    dash_for,
    enum_value,
    marker_for,
    resolve_class_representation,
    resolve_connection_edge_style,
    validate_representation,
)


class PlatformGenerator(GeneratorInterface):
    """
    PlatformGenerator creates a complete full-stack instance editor application.
    
    The generated application allows users to:
    - Create instances of classes defined in the input class diagram
    - Edit instance attributes
    - Create links between instances (associations)
    - Export instances as JSON or Python objects
    - Drag-and-drop interface similar to BESSER's class diagram editor
    
    Args:
        model (DomainModel): The B-UML domain model (class diagram) that defines
                            the metamodel for which instances will be created.
        customization (PlatformCustomizationModel, optional): Per-class and
                            per-association overrides (container flag, default
                            node size, edge color). None preserves the default
                            look for every class/association.
        output_dir (str, optional): Directory where generated code will be saved.
                                   Defaults to None (uses 'output' folder).

    Example:
        >>> from besser.BUML import DomainModel, Class, Property, StringType
        >>> from besser.BUML.metamodel.platform_customization import (
        ...     PlatformCustomizationModel, ClassCustomization, AssociationCustomization
        ... )
        >>>
        >>> # Define a simple class diagram
        >>> library_model = DomainModel(name="LibraryDomain")
        >>> Book = Class(name="Book")
        >>> Book.attributes = {Property(name="title", type=StringType)}
        >>> library_model.types = {Book}
        >>>
        >>> # Optional: customize how the generated editor renders Book instances
        >>> cust = PlatformCustomizationModel(
        ...     name="LibraryCustomization",
        ...     class_overrides={"Book": ClassCustomization(default_width=100, default_height=100)},
        ... )
        >>>
        >>> # Generate instance editor platform
        >>> generator = PlatformGenerator(library_model, customization=cust, output_dir="library_platform")
        >>> generator.generate()
    """

    def __init__(
        self,
        model: DomainModel,
        customization: Optional[PlatformCustomizationModel] = None,
        output_dir: str = None,
    ):
        super().__init__(model, output_dir)
        self.domain_model = model
        self.customization = customization
        
        # Setup Jinja2 environment
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=['jinja2.ext.do']
        )
        
        # Custom Jinja2 filters
        self.env.filters['safe_var_name'] = self._safe_var_name
        self.env.filters['python_type'] = self._get_python_type
        self.env.filters['ts_type'] = self._get_typescript_type
        self.env.filters['union'] = self._union_filter
        self.env.filters['dash_for'] = dash_for
        self.env.filters['marker_for'] = marker_for
        self.env.filters['enum_value'] = enum_value
        self.env.filters['indent_code'] = self._indent_code

    def _validate_method_bodies(self):
        """Parse every user-authored ``Method.code`` via ``ast.parse``
        and raise a targeted error if any of them is malformed Python.

        Catches the common authoring mistake of inconsistent indentation
        (mixed 2/4/6-space steps, or a line dedented below its block)
        at *generation time* instead of letting it leak into the emitted
        ``domain_classes.py`` — where the only signal is an obscure
        IndentationError when the operator tries to start the backend.

        Two body shapes are validated:
          * Full ``def name(...): ...`` — parsed directly (after
            ``textwrap.dedent`` to absorb any common leading whitespace).
          * Bare body — wrapped with a synthetic ``def _check(self, ...):``
            header that mirrors the method's parameters, then parsed.

        Empty / NONE-typed methods are skipped (the template emits a
        ``pass`` stub for those, which is always valid).
        """
        for cls in self.domain_model.get_classes():
            methods = getattr(cls, "methods", None) or set()
            for method in methods:
                impl = getattr(method, "implementation_type", None)
                code = (getattr(method, "code", "") or "").strip()
                if impl is None or impl.value != "code" or not code:
                    continue
                self._parse_method_or_raise(cls, method, code)

    @staticmethod
    def _parse_method_or_raise(cls, method, code: str) -> None:
        """Helper: parses one method's code and raises with class + method
        context if it fails. Kept static so the error message doesn't
        depend on PlatformGenerator state."""
        normalized = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        dedented = textwrap.dedent(normalized)
        if dedented.lstrip().startswith("def "):
            source_to_check = dedented
        else:
            # Wrap as a synthetic def using the method's declared params
            # so the indented body parses in the right scope.
            param_names = [
                getattr(p, "name", "") for p in (getattr(method, "parameters", []) or [])
            ]
            param_names = [p for p in param_names if p]
            signature = ", ".join(["self"] + param_names)
            body_indented = textwrap.indent(dedented, "    ")
            source_to_check = f"def _validate_check({signature}):\n{body_indented}\n"

        try:
            ast.parse(source_to_check)
        except SyntaxError as e:
            cls_name = getattr(cls, "name", "<unknown>")
            method_name = getattr(method, "name", "<unknown>")
            # Show the offending source so the user can see immediately
            # what they need to fix. Trim to the area around the error
            # so the message stays readable for long bodies.
            lines = source_to_check.splitlines()
            if e.lineno and 1 <= e.lineno <= len(lines):
                lo = max(0, e.lineno - 3)
                hi = min(len(lines), e.lineno + 2)
                snippet = "\n".join(
                    f"  {i + 1:4d} | {lines[i]}" for i in range(lo, hi)
                )
            else:
                snippet = ""
            raise ValueError(
                f"Python syntax error in method '{method_name}' on class '{cls_name}' "
                f"(line {e.lineno}, offset {e.offset}): {e.msg}.\n"
                f"\n{snippet}\n\n"
                f"Open the editor's Python Implementation panel for {cls_name}.{method_name} "
                f"and fix the indentation. Use 4 spaces per nested block, and make sure all "
                f"statements in the same block start at the same column."
            ) from e

    @staticmethod
    def _indent_code(code: str, spaces: int = 8) -> str:
        """Indent every line of a (possibly multi-line) user-authored method
        body so it sits inside the emitted ``def``.

        Pipeline:
          1. Normalize line endings (CRLF → LF) so platform-specific
             paste-from-Notepad payloads don't confuse the parser.
          2. Expand tabs to 4 spaces — tabs + spaces in the same body
             are a recipe for ``inconsistent use of tabs and spaces``
             at runtime.
          3. ``textwrap.dedent`` to strip any *common* leading whitespace
             so an author who copy-pasted from an already-indented file
             doesn't compound that offset with our re-indent.
          4. Re-indent every non-blank line by ``spaces``.

        Blank lines are preserved (Python tolerates them, and stripping
        would mess with line numbers in tracebacks).

        If the body is empty after stripping, returns ``"<spaces>pass"``
        so the resulting function is still syntactically valid.

        Note: this normalizes *leading* whitespace only. If the author's
        body has *internal* indentation inconsistencies (e.g. nested
        blocks at irregular depths), Python will still error at runtime —
        that's a real source bug, not a generator one.
        """
        if not code or not code.strip():
            return " " * spaces + "pass"
        # Normalize line endings + expand tabs before dedent so the
        # common-prefix calculation isn't thrown off by mixed whitespace.
        normalized = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        dedented = textwrap.dedent(normalized)
        return textwrap.indent(dedented, " " * spaces, predicate=lambda line: bool(line.strip()))

    def _representation_context(self):
        """Compute customization-derived metadata shared by several templates.

        Returns a dict with these keys:
          - ``class_representations``: name -> {'mode', 'port_side'}
          - ``port_classes_registry``: name -> {'portSide'}
          - ``connection_classes_registry``: name -> endpoint info
          - ``addable_port_classes``: name -> [{className, associationName}, …]
        """
        classes = list(self.domain_model.get_classes())
        registries = build_representation_registries(classes, self.customization)
        class_representations = {
            cls.name: resolve_class_representation(cls, self.customization)
            for cls in classes
        }
        # Resolve edge styling up the inheritance chain for any class that
        # ends up rendered as a connection — subclasses inherit the base
        # class's line/arrow/label settings unless they override them.
        connection_edge_styles = {}
        for cls in classes:
            rep = class_representations.get(cls.name)
            if rep and rep.get("mode") == "connection":
                merged = resolve_connection_edge_style(cls, self.customization)
                if merged is not None:
                    connection_edge_styles[cls.name] = merged
        subclass_registry = build_subclass_registry(classes)
        addable_port_classes = compute_addable_port_classes(
            classes,
            self.customization,
            subclass_registry,
            registries["port_classes"].keys(),
        )
        # Per-class invokable method registry. Drives the ▶ buttons in the
        # generated PropertyEditor — one per executable method. Only
        # implementation_type=CODE with a non-empty body is invokable today;
        # STATE_MACHINE/BAL show up in later phases via a different dispatcher
        # but are deliberately omitted here so the UI doesn't offer buttons
        # the runtime can't honour. Methods are sorted by name for
        # deterministic generation.
        class_methods_registry = {}
        for cls in classes:
            methods_list = []
            for method in sorted(
                getattr(cls, "methods", set()) or set(), key=lambda m: m.name
            ):
                impl = getattr(method, "implementation_type", None)
                code = (getattr(method, "code", "") or "").strip()
                if impl is None or impl.value != "code" or not code:
                    continue
                params = []
                for param in getattr(method, "parameters", []) or []:
                    p_type = getattr(getattr(param, "type", None), "name", "any")
                    params.append({"name": param.name, "type": p_type})
                return_type = (
                    getattr(getattr(method, "type", None), "name", None)
                    if getattr(method, "type", None) is not None
                    else None
                )
                methods_list.append(
                    {
                        "name": method.name,
                        "parameters": params,
                        "returnType": return_type,
                    }
                )
            if methods_list:
                class_methods_registry[cls.name] = methods_list

        # Classes whose tick-step method is flagged with is_step=True in the metadata.
        # The scheduler dispatches engine.tick(class, id, dt, method_name=...) on every
        # instance of these classes once per tick.
        steppable_classes = {}
        for cls in classes:
            for m in (getattr(cls, "methods", set()) or set()):
                if (
                    getattr(getattr(m, "metadata", None), "is_step", False)
                    and (getattr(m, "implementation_type", None) is not None)
                    and getattr(m, "implementation_type").value == "code"
                    and (getattr(m, "code", "") or "").strip()
                ):
                    steppable_classes[cls.name] = m.name
                    break

        # Per-class registry of attributes flagged as runtime input sources
        # via ``metadata.is_input = True``. The generated PropertyEditor
        # renders a simulator-source selector for each of these.
        input_attributes_registry = {}
        for cls in classes:
            input_attrs = [
                prop.name
                for prop in sorted(cls.attributes, key=lambda p: p.name)
                if getattr(getattr(prop, "metadata", None), "is_input", False)
            ]
            if input_attrs:
                input_attributes_registry[cls.name] = input_attrs

        return {
            "class_representations": class_representations,
            "port_classes_registry": registries["port_classes"],
            "connection_classes_registry": registries["connection_classes"],
            "connection_edge_styles": connection_edge_styles,
            "subclass_registry": subclass_registry,
            "addable_port_classes": addable_port_classes,
            "class_methods_registry": class_methods_registry,
            "steppable_classes": steppable_classes,
            "input_attributes_registry": input_attributes_registry,
        }

    def generate(self):
        """
        Generates the complete platform code including:
        - Backend (FastAPI with Python classes)
        - Frontend (React + TypeScript)
        - Docker configuration
        - Documentation
        """
        print(f"Generating instance editor platform for model: {self.domain_model.name}")

        # Validate platform customization (port / connection classes wiring)
        # before producing anything — otherwise the user gets a broken editor
        # with no clue what went wrong.
        if self.customization is not None:
            issues = validate_representation(
                list(self.domain_model.get_classes()), self.customization
            )
            if issues:
                bullet = "\n  - "
                raise ValueError(
                    "Platform customization has unresolved representation problems:"
                    + bullet
                    + bullet.join(issues)
                )

        # Catch broken Python in any user-authored method body BEFORE
        # writing files. Without this, an IndentationError in (say)
        # ``StorageTank.update_tank_level`` only surfaces when the
        # operator tries to start the generated backend — by which
        # point they're staring at a stack trace inside a generated
        # file they didn't write.
        self._validate_method_bodies()

        # Create directory structure
        self._create_directory_structure()
        
        # Generate backend
        self._generate_backend()
        
        # Generate frontend
        self._generate_frontend()
        
        # Generate Docker files
        self._generate_docker_files()
        
        # Generate documentation
        self._generate_docs()
        
        print(f"[OK] Platform generated successfully in: {self.output_dir}")
    
    def _create_directory_structure(self):
        """Creates the necessary directory structure for the platform."""
        dirs = [
            'backend',
            'backend/routers',
            'backend/services',
            'backend/metamodel',
            'backend/schemas',
            'frontend',
            'frontend/src',
            'frontend/src/components',
            'frontend/src/components/ui',
            'frontend/src/api',
            'frontend/src/lib',
            'frontend/src/types',
            'frontend/src/hooks',
            'frontend/src/utils',
            'docs'
        ]
        
        for dir_name in dirs:
            dir_path = os.path.join(self.output_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
    
    def _generate_backend(self):
        """Generates FastAPI backend code."""
        backend_dir = os.path.join(self.output_dir, 'backend')

        # Generate Python classes from the domain model
        self._generate_python_classes(backend_dir)

        # Generate FastAPI main application
        self._generate_backend_main(backend_dir)

        # Generate routers
        self._generate_backend_routers(backend_dir)

        # Generate services
        self._generate_backend_services(backend_dir)

        # Generate Pydantic schemas
        self._generate_backend_schemas(backend_dir)

        # Generate requirements.txt
        self._generate_backend_requirements(backend_dir)

        # Copy the runtime kernel verbatim. It's static Python (not Jinja)
        # because the engine is generic — wiring to per-domain class_map
        # happens in runtime/__init__.py via the singleton imports.
        self._copy_runtime_kernel(backend_dir)

    def _copy_runtime_kernel(self, backend_dir):
        """Copy ``besser/runtime/`` into ``<backend>/runtime/``.

        Single source of truth: the runtime kernel lives in BESSER core
        (so the web modeling editor can also import it for inline method
        execution on the object diagram). The platform generator still
        copies it verbatim so each deployed plant stays self-contained
        and doesn't depend on BESSER being installed on the server.

        After copying, we overwrite the kernel's ``__init__.py`` with a
        platform-flavoured bootstrap that builds the ``engine`` singleton
        from the generated ``services.instance_manager`` AND a generated
        ``associations.py`` that encodes the per-class association-end
        metadata (multiplicities, role names) so the engine can resolve
        the link graph at invocation time. Core's ``__init__.py``
        deliberately doesn't do any of this so the kernel is importable
        in contexts where ``services.instance_manager`` doesn't exist
        (the editor backend, tests, a REPL).
        """
        # besser/generators/platform/platform_generator.py → besser/runtime/
        # is three ``..`` hops: platform → generators → besser → runtime.
        here = os.path.dirname(os.path.abspath(__file__))
        besser_root = os.path.abspath(os.path.join(here, '..', '..'))
        src = os.path.join(besser_root, 'runtime')
        dst = os.path.join(backend_dir, 'runtime')
        # On regeneration the destination already exists; replace it.
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

        # Emit per-domain association-end metadata as static Python data.
        # The bootstrap reads this and feeds it to the Engine so method
        # bodies can navigate associations (``self.ports``,
        # ``port.incomingStreams``, ...) at invoke time. See Phase 1.0.8.
        from besser.runtime.materialize import build_associations_index
        assoc_index = build_associations_index(self.domain_model)
        association_entries = []
        seen = set()
        # Pull from both indices and de-duplicate — same end can land
        # in both ``by_end`` and ``by_assoc``.
        for meta in list(assoc_index.by_end.values()) + list(assoc_index.by_assoc.values()):
            key = (meta.owner_class, meta.end_name, meta.association_name)
            if key in seen:
                continue
            seen.add(key)
            association_entries.append({
                "owner_class": meta.owner_class,
                "end_name": meta.end_name,
                "target_class": meta.target_class,
                "multiplicity_max": meta.multiplicity_max,
                "association_name": meta.association_name,
            })
        associations_source = (
            '"""Per-class association-end metadata, generated from the\n'
            'source DomainModel. Drives runtime link resolution in the\n'
            'Engine (multiplicity-aware slot assignment + reverse-index\n'
            'bidirectional navigation).\n'
            '"""\n'
            'ASSOCIATION_ENDS = [\n'
        )
        for entry in association_entries:
            associations_source += '    ' + repr(entry) + ',\n'
        associations_source += ']\n'
        with open(os.path.join(dst, 'associations.py'), 'w', encoding='utf-8') as f:
            f.write(associations_source)

        # Build steppable_classes dict from the domain model — needed by the
        # scheduler bootstrap so it knows which instances to tick and which method to call.
        _steppable: dict = {}
        for _cls in self.domain_model.get_classes():
            for _m in (getattr(_cls, "methods", set()) or set()):
                if (
                    getattr(getattr(_m, "metadata", None), "is_step", False)
                    and (getattr(_m, "implementation_type", None) is not None)
                    and getattr(_m, "implementation_type").value == "code"
                    and (getattr(_m, "code", "") or "").strip()
                ):
                    _steppable[_cls.name] = _m.name
                    break
        steppable_classes_repr = repr(_steppable)

        # Override the copied __init__.py with the platform-side bootstrap.
        platform_init = (
            '"""Generated platform bootstrap for the besser4DT runtime kernel.\n'
            '\n'
            'Wires the generic Engine, Scheduler, InputBindings, and HistoryStore\n'
            'to this platform\'s singleton ``services.instance_manager`` and to\n'
            'the per-domain ``ASSOCIATION_ENDS`` table.\n'
            '"""\n'
            'from services.instance_manager import instance_manager\n'
            '\n'
            'from .associations import ASSOCIATION_ENDS\n'
            'from .bindings import InputBindings\n'
            'from .engine import Engine\n'
            'from .history import HistoryStore\n'
            'from .materialize import AssociationEndMeta, AssociationsIndex, materialize_classes\n'
            'from .scheduler import Scheduler\n'
            '\n'
            '_by_end = {}\n'
            '_by_assoc = {}\n'
            'for _entry in ASSOCIATION_ENDS:\n'
            '    _meta = AssociationEndMeta(**_entry)\n'
            '    if _meta.end_name:\n'
            '        _by_end[(_meta.owner_class, _meta.end_name)] = _meta\n'
            '    if _meta.association_name:\n'
            '        _by_assoc[(_meta.owner_class, _meta.association_name)] = _meta\n'
            '_associations_index = AssociationsIndex(by_end=_by_end, by_assoc=_by_assoc)\n'
            '\n'
            'engine = Engine(\n'
            '    instance_manager=instance_manager,\n'
            '    class_map=instance_manager.class_map,\n'
            '    associations_index=_associations_index,\n'
            ')\n'
            '\n'
            f'_steppable_classes = {steppable_classes_repr}\n'
            '\n'
            'history = HistoryStore(instance_manager=instance_manager)\n'
            'bindings = InputBindings(instance_manager=instance_manager)\n'
            'scheduler = Scheduler(\n'
            '    engine=engine,\n'
            '    history=history,\n'
            '    bindings=bindings,\n'
            '    instance_manager=instance_manager,\n'
            '    steppable_classes=_steppable_classes,\n'
            '    dt=1.0,\n'
            ')\n'
            '\n'
            '# Restore persisted bindings if they exist from a previous run.\n'
            'bindings.load("dt_bindings.json")\n'
            '\n'
            '__all__ = ["Engine", "engine", "materialize_classes",\n'
            '           "Scheduler", "scheduler", "InputBindings", "bindings",\n'
            '           "HistoryStore", "history"]\n'
        )
        with open(os.path.join(dst, '__init__.py'), 'w', encoding='utf-8') as f:
            f.write(platform_init)
    
    def _generate_python_classes(self, backend_dir):
        """Generates Python classes representing the domain model."""
        template = self.env.get_template('backend/models/domain_classes.py.j2')
        output_path = os.path.join(backend_dir, 'metamodel', 'domain_classes.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.classes_sorted_by_inheritance()
            ))
        
        # Generate __init__.py for metamodel
        init_template = self.env.get_template('backend/models/__init__.py.j2')
        init_path = os.path.join(backend_dir, 'metamodel', '__init__.py')
        
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(init_template.render(
                classes=self.domain_model.classes_sorted_by_inheritance()
            ))
    
    def _generate_backend_main(self, backend_dir):
        """Generates the main FastAPI application file."""
        template = self.env.get_template('backend/main.py.j2')
        output_path = os.path.join(backend_dir, 'main.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.get_classes()
            ))
    
    def _generate_backend_routers(self, backend_dir):
        """Generates FastAPI routers for CRUD operations."""
        routers_dir = os.path.join(backend_dir, 'routers')
        
        # Generate instance router (handles all instance operations)
        template = self.env.get_template('backend/routers/instances.py.j2')
        output_path = os.path.join(routers_dir, 'instances.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.classes_sorted_by_inheritance()
            ))
        
        # Generate export router
        template = self.env.get_template('backend/routers/export.py.j2')
        output_path = os.path.join(routers_dir, 'export.py')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(model=self.domain_model))

        # Generate runtime router (▶ Tick endpoint — Phase 1.0)
        template = self.env.get_template('backend/routers/runtime.py.j2')
        output_path = os.path.join(routers_dir, 'runtime.py')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(model=self.domain_model))

        # Generate runtime control router (Run/Pause/Step/Reset + /history + WS)
        template = self.env.get_template('backend/routers/runtime_control.py.j2')
        output_path = os.path.join(routers_dir, 'runtime_control.py')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(model=self.domain_model))

        # Generate input bindings router
        template = self.env.get_template('backend/routers/bindings.py.j2')
        output_path = os.path.join(routers_dir, 'bindings.py')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(model=self.domain_model))
    
    def _generate_backend_services(self, backend_dir):
        """Generates backend service layer."""
        services_dir = os.path.join(backend_dir, 'services')
        
        # Generate instance manager service
        template = self.env.get_template('backend/services/instance_manager.py.j2')
        output_path = os.path.join(services_dir, 'instance_manager.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.get_classes()
            ))
        
        # Generate export service
        template = self.env.get_template('backend/services/export_service.py.j2')
        output_path = os.path.join(services_dir, 'export_service.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(model=self.domain_model))
    
    def _generate_backend_schemas(self, backend_dir):
        """Generates Pydantic schemas for API requests/responses."""
        schemas_dir = os.path.join(backend_dir, 'schemas')
        
        template = self.env.get_template('backend/schemas/instance_schemas.py.j2')
        output_path = os.path.join(schemas_dir, 'instance_schemas.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.classes_sorted_by_inheritance()
            ))
    
    def _generate_backend_requirements(self, backend_dir):
        """Generates requirements.txt for backend dependencies."""
        template = self.env.get_template('backend/requirements.txt.j2')
        output_path = os.path.join(backend_dir, 'requirements.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
    
    def _generate_frontend(self):
        """Generates React + TypeScript frontend code."""
        frontend_dir = os.path.join(self.output_dir, 'frontend')
        
        # Generate package.json
        self._generate_frontend_package_json(frontend_dir)
        
        # Generate TypeScript types
        self._generate_frontend_types(frontend_dir)
        
        # Generate API client
        self._generate_frontend_api(frontend_dir)
        
        # Generate React components
        self._generate_frontend_components(frontend_dir)
        
        # Generate main App component
        self._generate_frontend_app(frontend_dir)
        
        # Generate index files
        self._generate_frontend_index(frontend_dir)
        
        # Generate configuration files
        self._generate_frontend_config(frontend_dir)
    
    def _generate_frontend_package_json(self, frontend_dir):
        """Generates package.json for the frontend."""
        template = self.env.get_template('frontend/package.json.j2')
        output_path = os.path.join(frontend_dir, 'package.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                project_name=self.domain_model.name.lower().replace(' ', '-')
            ))
    
    def _generate_frontend_types(self, frontend_dir):
        """Generates TypeScript type definitions."""
        types_dir = os.path.join(frontend_dir, 'src', 'types')

        template = self.env.get_template('frontend/src/types/models.ts.j2')
        output_path = os.path.join(types_dir, 'models.ts')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.classes_sorted_by_inheritance(),
                enumerations=sorted(
                    self.domain_model.get_enumerations(), key=lambda e: e.name
                ),
                customization=self.customization,
                **self._representation_context(),
            ))
    
    def _generate_frontend_api(self, frontend_dir):
        """Generates API client for backend communication."""
        api_dir = os.path.join(frontend_dir, 'src', 'api')
        
        # Generate base API client
        template = self.env.get_template('frontend/src/api/client.ts.j2')
        output_path = os.path.join(api_dir, 'client.ts')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate instances API
        template = self.env.get_template('frontend/src/api/instances.ts.j2')
        output_path = os.path.join(api_dir, 'instances.ts')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.get_classes()
            ))

        # Generate runtime control API (Run/Pause/Step/Reset/history/WS)
        template = self.env.get_template('frontend/src/api/runtime.ts.j2')
        output_path = os.path.join(api_dir, 'runtime.ts')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate bindings API (input source configuration)
        template = self.env.get_template('frontend/src/api/bindings.ts.j2')
        output_path = os.path.join(api_dir, 'bindings.ts')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
    
    def _generate_frontend_components(self, frontend_dir):
        """Generates React components."""
        components_dir = os.path.join(frontend_dir, 'src', 'components')
        
        rep_ctx = self._representation_context()

        # Generate InstanceCanvas component (main editor)
        template = self.env.get_template('frontend/src/components/InstanceCanvas.tsx.j2')
        output_path = os.path.join(components_dir, 'InstanceCanvas.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.get_classes(),
                customization=self.customization,
                **rep_ctx,
            ))

        # Generate InstanceNode component (visual representation of instances)
        template = self.env.get_template('frontend/src/components/InstanceNode.tsx.j2')
        output_path = os.path.join(components_dir, 'InstanceNode.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                customization=self.customization,
                **rep_ctx,
            ))
        
        # Generate PropertyEditor component
        template = self.env.get_template('frontend/src/components/PropertyEditor.tsx.j2')
        output_path = os.path.join(components_dir, 'PropertyEditor.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                input_attributes_registry=rep_ctx["input_attributes_registry"],
                class_methods_registry=rep_ctx["class_methods_registry"],
            ))

        # Generate HistoryChart component (Recharts time-series for is_input attrs)
        template = self.env.get_template('frontend/src/components/HistoryChart.tsx.j2')
        output_path = os.path.join(components_dir, 'HistoryChart.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate ClassPalette component (for dragging classes to create instances)
        template = self.env.get_template('frontend/src/components/ClassPalette.tsx.j2')
        output_path = os.path.join(components_dir, 'ClassPalette.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                classes=self.domain_model.get_classes()
            ))

        # Generate AssociationSelector component
        template = self.env.get_template('frontend/src/components/AssociationSelector.tsx.j2')
        output_path = os.path.join(components_dir, 'AssociationSelector.tsx')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate InstanceCreationModal component
        template = self.env.get_template('frontend/src/components/InstanceCreationModal.tsx.j2')
        output_path = os.path.join(components_dir, 'InstanceCreationModal.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate AddPortPopover component (the hover-revealed "+" affordance
        # on equipment nodes that spawns a port via composition).
        template = self.env.get_template('frontend/src/components/AddPortPopover.tsx.j2')
        output_path = os.path.join(components_dir, 'AddPortPopover.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate AddAssociationPopover component (the hover-revealed "↔"
        # affordance for picking / creating association targets — replaces the
        # old drag-from-side-handle UX as the primary association entry point).
        template = self.env.get_template('frontend/src/components/AddAssociationPopover.tsx.j2')
        output_path = os.path.join(components_dir, 'AddAssociationPopover.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate StreamClassSelector component (the modal shown when a
        # port-to-port drag matches multiple connection-class candidates so the
        # user can pick which concrete class to instantiate).
        template = self.env.get_template('frontend/src/components/StreamClassSelector.tsx.j2')
        output_path = os.path.join(components_dir, 'StreamClassSelector.tsx')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate shared lib (cn helper)
        lib_dir = os.path.join(frontend_dir, 'src', 'lib')
        template = self.env.get_template('frontend/src/lib/utils.ts.j2')
        with open(os.path.join(lib_dir, 'utils.ts'), 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate shadcn/ui primitive components (static; wrapped in {% raw %})
        ui_dir = os.path.join(components_dir, 'ui')
        ui_components = ['button', 'input', 'label', 'card', 'dialog']
        for comp in ui_components:
            template = self.env.get_template(f'frontend/src/components/ui/{comp}.tsx.j2')
            with open(os.path.join(ui_dir, f'{comp}.tsx'), 'w', encoding='utf-8') as f:
                f.write(template.render())

    def _generate_frontend_app(self, frontend_dir):
        """Generates the main App component."""
        template = self.env.get_template('frontend/src/App.tsx.j2')
        output_path = os.path.join(frontend_dir, 'src', 'App.tsx')
        rep_ctx = self._representation_context()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                customization=self.customization,
                steppable_classes=rep_ctx["steppable_classes"],
                input_attributes_registry=rep_ctx["input_attributes_registry"],
            ))
    
    def _generate_frontend_index(self, frontend_dir):
        """Generates index.html and index.tsx."""
        # Generate index.html (must be in root for Vite)
        template = self.env.get_template('frontend/index.html.j2')
        output_path = os.path.join(frontend_dir, 'index.html')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                project_name=self.domain_model.name
            ))
        
        # Generate index.tsx
        template = self.env.get_template('frontend/src/index.tsx.j2')
        output_path = os.path.join(frontend_dir, 'src', 'index.tsx')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate vite-env.d.ts
        template = self.env.get_template('frontend/src/vite-env.d.ts.j2')
        output_path = os.path.join(frontend_dir, 'src', 'vite-env.d.ts')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate index.css
        template = self.env.get_template('frontend/src/index.css.j2')
        output_path = os.path.join(frontend_dir, 'src', 'index.css')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
    
    def _generate_frontend_config(self, frontend_dir):
        """Generates frontend configuration files."""
        # Generate tsconfig.json
        template = self.env.get_template('frontend/tsconfig.json.j2')
        output_path = os.path.join(frontend_dir, 'tsconfig.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate tsconfig.node.json
        template = self.env.get_template('frontend/tsconfig.node.json.j2')
        output_path = os.path.join(frontend_dir, 'tsconfig.node.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate vite.config.ts
        template = self.env.get_template('frontend/vite.config.ts.j2')
        output_path = os.path.join(frontend_dir, 'vite.config.ts')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate tailwind.config.ts
        template = self.env.get_template('frontend/tailwind.config.ts.j2')
        output_path = os.path.join(frontend_dir, 'tailwind.config.ts')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

        # Generate postcss.config.js
        template = self.env.get_template('frontend/postcss.config.js.j2')
        output_path = os.path.join(frontend_dir, 'postcss.config.js')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())

    def _generate_docker_files(self):
        """Generates Docker and docker-compose configuration."""
        # Generate docker-compose.yml
        template = self.env.get_template('docker-compose.yml.j2')
        output_path = os.path.join(self.output_dir, 'docker-compose.yml')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                project_name=self.domain_model.name.lower().replace(' ', '-')
            ))
        
        # Generate backend Dockerfile
        template = self.env.get_template('backend/Dockerfile.j2')
        output_path = os.path.join(self.output_dir, 'backend', 'Dockerfile')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
        
        # Generate frontend Dockerfile
        template = self.env.get_template('frontend/Dockerfile.j2')
        output_path = os.path.join(self.output_dir, 'frontend', 'Dockerfile')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render())
    
    def _generate_docs(self):
        """Generates README and documentation."""
        template = self.env.get_template('README.md.j2')
        output_path = os.path.join(self.output_dir, 'README.md')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                classes=self.domain_model.get_classes()
            ))
    
    # Helper methods
    
    @staticmethod
    def _safe_var_name(name: str) -> str:
        """Convert a name to a safe Python/JavaScript variable name."""
        # Replace spaces and special characters with underscores
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        # Ensure it doesn't start with a number
        if safe_name and safe_name[0].isdigit():
            safe_name = '_' + safe_name
        return safe_name or '_unnamed'
    
    @staticmethod
    def _get_python_type(buml_type) -> str:
        """Convert BUML type to Python type hint."""
        from besser.BUML.metamodel.structural import PrimitiveDataType
        
        if isinstance(buml_type, PrimitiveDataType):
            type_map = {
                'int': 'int',
                'integer': 'int',
                'float': 'float',
                'double': 'float',
                'string': 'str',
                'str': 'str',
                'bool': 'bool',
                'boolean': 'bool',
                'date': 'date',
                'time': 'time',
                'datetime': 'datetime',
            }
            return type_map.get(buml_type.name.lower(), 'Any')
        return 'Any'
    
    @staticmethod
    def _get_typescript_type(buml_type) -> str:
        """Convert BUML type to TypeScript type."""
        from besser.BUML.metamodel.structural import PrimitiveDataType
        
        if isinstance(buml_type, PrimitiveDataType):
            type_map = {
                'int': 'number',
                'integer': 'number',
                'float': 'number',
                'double': 'number',
                'string': 'string',
                'str': 'string',
                'bool': 'boolean',
                'boolean': 'boolean',
                'date': 'string',
                'time': 'string',
                'datetime': 'string',
            }
            return type_map.get(buml_type.name.lower(), 'any')
        return 'any'
    
    @staticmethod
    def _union_filter(set1, set2):
        """Custom Jinja2 filter to combine two sets."""
        if set1 is None:
            set1 = set()
        if set2 is None:
            set2 = set()
        # Convert to sets if they're not already
        s1 = set(set1) if not isinstance(set1, set) else set1
        s2 = set(set2) if not isinstance(set2, set) else set2
        return s1 | s2
