"""
Platform Generator for BESSER

Generates a complete instance editor platform that allows users to create
instances of classes defined in a B-UML class diagram. The generated platform
includes a FastAPI backend and React + TypeScript frontend similar to the
BESSER web modeling editor.
"""

import os
import shutil
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

    @staticmethod
    def _indent_code(code: str, spaces: int = 8) -> str:
        """Indent every line of a (possibly multi-line) user-authored method
        body so it sits inside the emitted ``def``. Empty trailing lines are
        preserved verbatim — Python tolerates them and stripping would mess
        with line numbers in tracebacks. If the body is blank, returns
        ``" " * spaces + "pass"`` so the resulting function is still valid.
        """
        if not code or not code.strip():
            return " " * spaces + "pass"
        prefix = " " * spaces
        return "\n".join(prefix + line if line else line for line in code.splitlines())

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
        # Per-class flag: does the class declare an executable ``step(self, dt)``
        # method? Drives the ▶ Tick affordance in the generated UI — we only
        # render it for classes the runtime kernel can actually tick. Methods
        # with implementation_type other than CODE are skipped here (Phase 1.0
        # only executes Python bodies; STATE_MACHINE/BAL come later).
        tickable_classes = {}
        for cls in classes:
            for method in getattr(cls, "methods", set()) or set():
                if (
                    getattr(method, "name", "") == "step"
                    and getattr(method, "implementation_type", None) is not None
                    and method.implementation_type.value == "code"
                    and (method.code or "").strip()
                ):
                    tickable_classes[cls.name] = True
                    break
        return {
            "class_representations": class_representations,
            "port_classes_registry": registries["port_classes"],
            "connection_classes_registry": registries["connection_classes"],
            "connection_edge_styles": connection_edge_styles,
            "subclass_registry": subclass_registry,
            "addable_port_classes": addable_port_classes,
            "tickable_classes": tickable_classes,
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
        
        print(f"✓ Platform generated successfully in: {self.output_dir}")
    
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
        """Copy besser/generators/platform/runtime/ into <backend>/runtime/.

        Static Python — no templating. The Engine is generic and gets
        bootstrapped against the generated ``services.instance_manager``
        via plain imports in ``runtime/__init__.py``.
        """
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime')
        dst = os.path.join(backend_dir, 'runtime')
        # On regeneration the destination already exists; replace it.
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    
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

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template.render(
                model=self.domain_model,
                customization=self.customization,
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
