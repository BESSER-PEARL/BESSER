"""
MATLAB GUI Generator
Generates a MATLAB uifigure-based GUI (.m file) from a BESSER GUIModel.
"""
import logging
import os
import re
import tempfile
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface

logger = logging.getLogger(__name__)

LEAF_TYPES = {
    'text', 'button', 'action-button', 'input', 'form',
    'image', 'link', 'link-button', 'menu', 'data-list',
    'embedded-content', 'table', 'bar-chart', 'line-chart',
    'pie-chart', 'radar-chart', 'radial-bar-chart',
    'metric-card', 'agent-component',
}


def flatten_components(components: list) -> list:
    """Recursively flatten containers, returning only leaf widgets."""
    result = []
    for comp in (components or []):
        ctype = (comp.get('type') or 'component').lower()
        if ctype in LEAF_TYPES:
            result.append(comp)
        else:
            result.extend(flatten_components(comp.get('children') or []))
    return result


def safe_id(value: str, prefix: str = 'c') -> str:
    """
    Convert any string to a safe MATLAB identifier.
    - Replace all non-alphanumeric chars with _
    - Ensure it never starts with a digit
    - Prefix with given prefix
    """
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', str(value or 'x'))
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if not cleaned or cleaned[0].isdigit():
        cleaned = prefix + '_' + cleaned
    return cleaned


class MatlabGuiGenerator(GeneratorInterface):

    def __init__(self, model: DomainModel,
                 gui_model=None,
                 output_dir: str = None):
        super().__init__(model, output_dir)
        self.gui_model = gui_model

    def generate(self) -> str:
        app_name = safe_id(getattr(self.model, 'name', 'GeneratedApp'), 'App')

        file_name = f"{app_name}App.m"
        file_path = self.build_generation_path(file_name=file_name)

        templates_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'templates'
        )
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Expose helpers inside templates
        env.globals['flatten']  = flatten_components
        env.globals['safe_id']  = safe_id

        pages = self._extract_pages()
        logger.debug('MatlabGuiGenerator: %d page(s)', len(pages))

        template = env.get_template('matlab_gui_template.m.j2')
        code = template.render(
            app_name=app_name,
            model=self.model,
            pages=pages,
        )

        with open(file_path, mode='w', encoding='utf-8') as fh:
            fh.write(code)

        logger.debug('MATLAB GUI written to: %s', file_path)
        return file_path

    def _extract_pages(self) -> list:
        if not self.gui_model:
            logger.warning('MatlabGuiGenerator: no gui_model provided')
            return []
        try:
            from besser.generators.react.react import ReactGenerator
            with tempfile.TemporaryDirectory() as tmp:
                rg = ReactGenerator(
                    model=self.model,
                    gui_model=self.gui_model,
                    output_dir=tmp,
                )
                payload, _, _ = rg._serialize_gui_model()
                pages = payload.get('pages', [])
                logger.debug('Serialized %d page(s)', len(pages))
                return pages
        except Exception as exc:
            logger.warning('Could not extract pages: %s', exc)
            return []