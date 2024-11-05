"""
This script converts a Draw.io XML file to a BUML model.
Modules:
    drawio_xml_to_buml (as buml_generator): Module to convert Draw.io XML to BUML.
    besser.BUML.notations.structuralPlantUML: Module to convert PlantUML to BUML.
    besser.BUML.metamodel.structural: Module containing the DomainModel class.
Usage:
    Specify the path to the XML file you want to analyze by setting the `xml_file` variable.
    Call the `xml_to_buml` function from the `drawio_xml_to_buml` module with the XML file as a parameter.
    The generated BUML model will be stored in the `buml_model` variable.
Example:
"""

import besser.BUML.notations.structuralDrawioXML.drawio_xml_to_buml as buml_generator
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.metamodel.structural import DomainModel

# Specify the path to the XML file you want to analyze
xml_file = 'model.drawio'

# Convert the XML file to a BUML model
buml_model = buml_generator.xml_to_buml(xml_file)