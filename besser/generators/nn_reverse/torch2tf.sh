#!/bin/bash


# Get the BUML code
python torch2buml/torch2buml.py $1


# Modify the code generator to generate tf code instead of torch code (default)
sed -i 's/pytorch/tf/g; s/Pytorch/TF/g' code_transformed.py

# Run the code generator
python code_transformed.py

