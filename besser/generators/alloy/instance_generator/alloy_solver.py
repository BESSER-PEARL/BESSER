import logging
import os
import tempfile
from besser.BUML.metamodel.structural import DomainModel
from besser.generators.alloy.alloy_generator import AlloyGenerator
from besser.generators.alloy.instance_generator.alloy_analyzer_executor import AlloyAnalyzerExecutor, AlloyResult
from besser.generators.alloy.instance_generator.alloy_instance_to_BUML import AlloyToBUML
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code

logger = logging.getLogger(__name__)

class AlloySolver:
    """Performs different kinds of automated analyses on B-UML models annotated with OCL invarints. 
    It can check for model consistency (i.e., satisfiability) and generate object diagrams for the model. 
    It employs the Alloy Analyzer as a backend for analysis."""

    def __init__(self, model: DomainModel, output_dir: str | None = None, scope: int = 5):
        if output_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="alloy_")
            output_dir = self._temp_dir.name
        self.scope = scope
        self.model = model
        self.output_dir = output_dir
        self.alloy_output_dir = os.path.join(self.output_dir, "alloy_output")
        generator = AlloyGenerator(model=self.model, output_dir=self.output_dir, scope=self.scope)
        generator.generate()
        self.specification = os.path.join(self.output_dir, "model.als")
        self.executor = AlloyAnalyzerExecutor()

    def check_consistency(self) -> AlloyResult:
        """Execute the Alloy Analyzer and check model satisfiability.
        Returns an AlloyResult indicating whether the model is satisfiable 
        (SAT), unsatisfiable (UNSAT), or if the analysis timed out (TIMEOUT).
        """
        (result, instance_xml_files) = self.executor.generate_instances(self.specification, self.alloy_output_dir)
        return result

    def generate_object_diagrams(self, num_instances: int = 1):
        """Generates BUML object diagrams using Alloy.
        Returns an AlloyResult indicating the result of the analysis and a list of 
        BUML instances. The list is empty if no satisfying instances were found or if 
        the analysis timed out.

        Args:
            num_instances: Number of satisfying instances to request from the Alloy
                Analyzer.
        """
        for_editor = True
        (res, instance_xml_files) = self.executor.generate_instances(self.specification, 
                                            self.alloy_output_dir, num_instances=num_instances)
        buml_instances = []
        for xml_path in instance_xml_files:
            converter = AlloyToBUML(xml_path)
            buml_instances.append(converter.generate_object_diagram(for_editor=for_editor))

        return (res, buml_instances)

    def generate_class_and_object_model(self):
        """Generates an object diagrams from the Alloy specification and combines it with 
        the class diagram to produce a complete BUML model code. 
        Returns the generated BUML model code in file ``output_dir/buml_class_object_model.py``.

        The object model section is emitted in the "editor" dialect so the
        generated file can be re-imported into the web editor with the
        objects' relationships (ObjectLinks) intact.
        """
        (res, buml_instances) = self.generate_object_diagrams(num_instances=1)
        if res == AlloyResult.UNSAT:
            return AlloyResult.UNSAT

        outfile = os.path.join(self.output_dir, "buml_class_object_model.py")
        # Clean up previous file before writing a new one
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(outfile):
            os.remove(outfile)

        # Write generated BUML model code to outfile
        domain_model_to_code(self.model, file_path=outfile)

        # Write generated BUML object model to outfile
        with open(outfile, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("\n################\n")
            f.write("# OBJECT MODEL #\n")
            f.write("################\n")
            f.write("\n")
            f.write(buml_instances[0])
            f.write("\n")

            # Write a generic project to outfile
            f.write("\n######################\n")
            f.write("# PROJECT DEFINITION #\n")
            f.write("######################\n")
            f.write("\n")
            f.write("from besser.BUML.metamodel.project import Project\n")
            f.write("\n")
            f.write("project = Project(\n")
            f.write('\tname="Alloy_Instance_Project", \n')
            f.write("\tmodels=[domain_model, object_model]\n")
            f.write(")\n")

        return AlloyResult.SAT 

