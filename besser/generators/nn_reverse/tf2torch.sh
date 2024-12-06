#!/bin/bash


# Get the BUML code
python tf2buml/tf2buml.py $1 # --datashape $2 

# Modify the code generator to generate torch code instead of tf code (default)
sed -i "s/tf/pytorch/g; s/TF/Pytorch/g" code_transformed.py


# Run the code generator --datashape $2 NeuralNetwork my_model
python code_transformed.py
