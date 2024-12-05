"""
Script to convert TensorFlow code to BUML code.

Argument:
    --filename (str): Path to a TensorFlow file containing 
        the code to transform.
"""

import sys
from besser.generators.nn_reverse.tf2buml.ast_parser_tf import ASTParserTF
from besser.generators.nn_reverse.tf2buml.transform_func_tf import (
    wrap_transform_layers
)
from besser.generators.nn_reverse.tf2buml.definitions import config_list, \
    train_param_list, test_param_list, lookup_loss_func
from besser.generators.nn_reverse.code2buml.utils_code2buml import (
    parse_arguments_code2buml, code2buml
)


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
