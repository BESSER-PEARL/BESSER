#!/bin/bash


python torch2buml.py ../output/$1/pytorch_nn.py  
python torch_transformed.py

diff output/NeuralNetwork/pytorch_nn.py ../output/$1/pytorch_nn.py


