#!/bin/bash


python tf2buml.py ../output/$1/tf_nn.py  
python tf_transformed.py

diff output/my_model/tf_nn.py ../output/$1/tf_nn.py


