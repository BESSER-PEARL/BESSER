"""
Script to convert TensorFlow code to BUML code.

Argument:
    --filename (str): Path to a TensorFlow file containing 
        the code to transform.
"""

import sys
from tests.nn.tf2buml.ast_parser import ASTParserTF
from tests.nn.tf2buml.transform_functions import wrap_transform_layers
from tests.nn.tf2buml.definitions import config_list, train_param_list, \
    test_param_list, lookup_loss_func
from tests.nn.code2buml.utils import parse_arguments_code2buml, code2buml



def main():
    """Tt transforms TF code to BUML code"""
    args = parse_arguments_code2buml()

    f = open("code_transformed.py", "w", encoding="utf-8")
    sys.stdout = f

    nn_name = code2buml(args, ASTParserTF, "TF", wrap_transform_layers,
              config_list, train_param_list, test_param_list,
              lookup_loss_func)

    print(f"tf_model = TFGenerator(model={nn_name}, "
          f"output_dir='output/{nn_name}')")
    print("tf_model.generate()")

    sys.stdout = sys.__stdout__
    f.close()

if __name__ == "__main__":
    main()
