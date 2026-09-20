"""
BUML model / object diagram integration for UML-BESSER models.

This module contains ``BUMLModelIntegrator``, which integrates an original
BUML model (class diagram) with an object diagram generated from Alloy,
producing a single, directly importable/executable BUML source file.
"""

import re

from besser.generators.alloy.instance_generator.alloy_instance_to_BUML import (
    AlloyToBUML,
)


class BUMLModelIntegrator:
    """Integrates an original BUML model with an object diagram generated from Alloy."""

    def __init__(self, original_buml_content: str, xml_instance_file: str):
        """
        Initializes the integrator.

        Args:
            original_buml_content: Source code of the original BUML file (class diagram)
            xml_instance_file: Path to the Alloy instance XML file
        """
        self.xml_instance_file = xml_instance_file
        self.original_content = original_buml_content

    def extract_structural_model_section(self) -> str:
        """
        Extracts the structural model section (class diagram) from the original BUML content.

        Returns:
            The code of the structural model section.
        """
        patterns = [
            r'################\s*\n#\s*OBJECT MODEL\s*#',
            r'##############\s*\n\s*from besser\.BUML\.metamodel\.object',
            r'######################\s*\n#\s*PROJECT DEFINITION\s*#'
        ]

        end_pos = len(self.original_content)
        for pattern in patterns:
            match = re.search(pattern, self.original_content, re.IGNORECASE)
            if match:
                end_pos = min(end_pos, match.start())

        structural_section = self.original_content[:end_pos].rstrip()
        return structural_section

    def extract_project_section(self) -> str:
        """
        Extracts the project definition section from the original BUML content if it exists.

        Returns:
            The code of the project section or an empty string if not found.
        """
        pattern = r'######################\s*\n#\s*PROJECT DEFINITION\s*#\s*\n######################\s*\n(.*)'
        match = re.search(pattern, self.original_content, re.DOTALL)

        if match:
            project_section = match.group(0).strip()

            models_pattern = r'(models=\[)([^\]]+)(\])'

            def replace_models(match):
                prefix = match.group(1)
                models_list = match.group(2).strip()
                suffix = match.group(3)

                if 'object_model' in models_list:
                    return match.group(0)

                if models_list:
                    return f"{prefix}{models_list}, object_model{suffix}"
                return f"{prefix}object_model{suffix}"

            project_section = re.sub(models_pattern, replace_models, project_section)

            return project_section
        return ""

    def _build_default_project_section(self) -> str:
        """
        Builds a default ``PROJECT DEFINITION`` section wrapping the structural
        and object models.

        Without this section, the integrated file would be re-imported as a
        bare ClassDiagram, whose parser strips ``import`` statements (assuming
        they are unnecessary) but not the ``datetime.date(...)`` calls used by
        the object model section, causing a ``NameError``. Wrapping both
        models in a ``Project`` makes the importer split and parse each
        section with its dedicated (datetime-safe) converter instead.

        Returns:
            The code of the project definition section.
        """
        return "\n".join([
            "######################",
            "# PROJECT DEFINITION #",
            "######################",
            "",
            "from besser.BUML.metamodel.project import Project",
            "from besser.BUML.metamodel.structural.structural import Metadata",
            "",
            'metadata = Metadata(description="Project generated from an Alloy-consistent instance.")',
            "project = Project(",
            '    name="Alloy_Instance_Project",',
            "    models=[domain_model, object_model],",
            '    owner="BESSER User",',
            "    metadata=metadata",
            ")",
        ])

    def generate_integrated_model(self, output_file: str | None = None) -> str:
        """
        Generates the complete integrated BUML model.

        Args:
            output_file: File to save the model (optional)

        Returns:
            The code of the integrated model.
        """
        structural_section = self.extract_structural_model_section()

        converter = AlloyToBUML(self.xml_instance_file)
        object_diagram_code = converter.generate_object_diagram(
            date_as_datetime=True, for_editor=False
        )

        project_section = self.extract_project_section()
        if not project_section:
            project_section = self._build_default_project_section()

        integrated_lines = []

        integrated_lines.append(structural_section)
        integrated_lines.append("")
        integrated_lines.append("")

        integrated_lines.append("################")
        integrated_lines.append("# OBJECT MODEL #")
        integrated_lines.append("################")
        integrated_lines.append("")
        integrated_lines.append(object_diagram_code)
        integrated_lines.append("")
        integrated_lines.append("")

        integrated_lines.append(project_section)

        integrated_model = "\n".join(integrated_lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(integrated_model)

        return integrated_model
