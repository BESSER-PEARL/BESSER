"""Module for launching the conversion of Draw.io XML files to BUML models."""

import os
from besser.BUML.notations.structuralDrawIO import structural_drawio_to_buml

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the path to your .drawio file
FILE = os.path.join(current_dir, 'Library.drawio')

# Convert the drawio file to a BUML model
buml_model = structural_drawio_to_buml(FILE, buml_model_file_name='buml_model')
