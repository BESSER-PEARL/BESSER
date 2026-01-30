"""
React generator entry point.
Splits rendering and serialization logic into mixins for clarity.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.gui import GUIModel
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

from .page_builder import PageBuilderMixin
from .serialization import GuiSerializationMixin


class ReactGenerator(GuiSerializationMixin, PageBuilderMixin, GeneratorInterface):
    """
    Generates React code based on BUML and GUI models.
    Walks templates, renders them with serialized GUI metadata, and builds pages.
    """

    def __init__(self, model: DomainModel, gui_model: GUIModel, output_dir: str = None):
        super().__init__(model, output_dir)
        self.gui_model = gui_model
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do"],
            variable_start_string="[[",
            variable_end_string="]]",
        )
        self._style_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self._raw_style_entries: List[Dict[str, Any]] = list(getattr(self.gui_model, "_style_entries", []))

    def generate(self):
        """
        Generates React TS code based on the provided BUML and GUI models.
        Generates all files from the templates directory, preserving structure and file names (removing .j2 extension).
        Generates all chart component files (ChartBlock imports them unconditionally).
        """

        context = self._build_generation_context()
        used_component_types = self._get_used_component_types()

        def generate_file_from_template(rel_template_path: str):
            if rel_template_path.endswith(".j2"):
                rel_output_path = rel_template_path[:-3]
            else:
                rel_output_path = rel_template_path

            file_path = self.build_generation_path(file_name=rel_output_path)
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                template = self.env.get_template(rel_template_path.replace("\\", "/"))
                generated_code = template.render(**context)
                with open(file_path, mode="w", encoding="utf-8") as f:
                    f.write(generated_code)
            except Exception as exc:
                print(f"Error generating {file_path} from {rel_template_path}: {exc}")
                raise

        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        for root, _, files in os.walk(templates_path):
            for file_name in files:
                abs_template_path = os.path.join(root, file_name)
                rel_template_path = os.path.relpath(abs_template_path, templates_path)

                if not self._should_generate_file(rel_template_path, used_component_types):
                    continue

                if file_name.endswith(".j2"):
                    generate_file_from_template(rel_template_path)
                else:
                    rel_output_path = rel_template_path
                    dest_path = self.build_generation_path(file_name=rel_output_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(abs_template_path, dest_path)

        self._generate_pages()

    def _should_generate_file(self, rel_path: str, used_component_types: set) -> bool:
        """Determine if a file should be generated based on component usage."""
        if "charts" + os.sep not in rel_path and "table" + os.sep not in rel_path:
            return True

        if "charts" + os.sep in rel_path:
            return True

        component_template_map = {
            "LineChart": ["LineChartComponent.tsx.j2"],
            "BarChart": ["BarChartComponent.tsx.j2"],
            "PieChart": ["PieChartComponent.tsx.j2"],
            "RadarChart": ["RadarChartComponent.tsx.j2"],
            "RadialBarChart": ["RadialBarChartComponent.tsx.j2"],
            "Table": [
                "TableComponent.tsx.j2",
                "TableComponent.css.j2",
                "ColumnFilter.tsx.j2",
                "ColumnSort.tsx.j2",
            ],
            "MetricCard": ["MetricCardComponent.tsx.j2"],
        }

        file_name = os.path.basename(rel_path)
        for component_type, template_files in component_template_map.items():
            if file_name in template_files:
                return component_type in used_component_types

        return True
