"""
It converts PyTorch code to BUML code.

Argument:
    filename (str): Path to a TensorFlow file containing 
        the code to transform.
    configfile (str, optional): Path to the configuration file that
        has the values of 'input_nn_type', 'output_nn_type', and 'only_nn'.
    datashape (str): 
"""

from besser.generators.nn_migration.torch2tf.ast_parser_pytorch import (
    ASTParserTorch
)
from besser.generators.nn_migration.transform_code import (
    parse_arguments_transform, transform
)
from besser.generators.nn.tf.tf_code_generator import TFGenerator


def main():
    """Tt transforms PyTorch code to TensorFlow code"""
    args = parse_arguments_transform()

    buml_model, output_nn_type = transform(args, "PyTorch", ASTParserTorch)

    tf_model = TFGenerator(
        model=buml_model, output_dir="output/test_new",
        generation_type=output_nn_type
    )
    tf_model.generate()

if __name__ == "__main__":
    main()
