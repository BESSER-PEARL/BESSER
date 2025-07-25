"""
It converts TensorFlow code to BUML code.

Argument:
    filename (str): Path to a TensorFlow file containing 
        the code to transform.
    configfile (str, optional): Path to the configuration file that
        has the values of 'input_nn_type', 'output_nn_type', and 'only_nn'.
    datashape (str, optional): The shape of the input data (optional). 
        It is needed when transforming tf code to pytorch code as it is
        used to recover some layer attributes dynamically. If the migrated 
        script defines a dataset, it can be skipped.
"""

from besser.generators.nn_migration.tf2torch.ast_parser_tf import ASTParserTF

from besser.generators.nn_migration.transform_code import (
    parse_arguments_transform, transform
)
from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator

def main():
    """Tt transforms TensorFlow code to PyTorch code"""
    args = parse_arguments_transform()

    buml_model, output_nn_type = transform(args, "TF", ASTParserTF)

    pytorch_model = PytorchGenerator(
        model=buml_model, output_dir="output/migrated_nn",
        generation_type=output_nn_type, channel_last=True
    )
    pytorch_model.generate()



if __name__ == "__main__":
    main()
