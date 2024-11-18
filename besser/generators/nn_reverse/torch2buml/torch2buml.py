"""
Script to convert PyTorch code to BUML code.

Argument:
    --filename (str): Path to a PyTorch file containing 
        the code to transform.
"""

import sys
from tests.nn.torch2buml.ast_parser import ASTParserTorch
from tests.nn.torch2buml.transform_functions import transform_layers
from tests.nn.torch2buml.definitions import config_list, train_param_list, \
    test_param_list, lookup_loss_func
from tests.nn.code2buml.utils import parse_arguments_code2buml, code2buml



def main():
    """Tt transforms PyTorch code to BUML code"""
    args = parse_arguments_code2buml()

    f = open("code_transformed.py", "w", encoding="utf-8")
    sys.stdout = f

    nn_name = code2buml(args, ASTParserTorch, "Pytorch", transform_layers,
                        config_list, train_param_list, test_param_list,
                        lookup_loss_func)

    print(f"pytorch_model = PytorchGenerator(model={nn_name}, "
          f"output_dir='output/{nn_name}')")
    print("pytorch_model.generate()")

    sys.stdout = sys.__stdout__
    f.close()

if __name__ == "__main__":
    main()
