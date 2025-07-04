"""
Config file for pytest test script.
"""

def pytest_addoption(parser):
    """
    Gets args to test file.
    """
    parser.addoption("--ftensorflow", required=True, action="store",
                     help="Path to TensorFlow NN file")
    parser.addoption("--fpytorch", required=False, action="store",
                     help="Path to PyTorch NN file")
    parser.addoption("--datashape", required=False, action="store",
                     type=parse_tuple,
                     help="The shape of the input data.")



def parse_tuple(value: str):
    """
    Parse a tuple from a string input
    
    Parameters:
    ----------
    value(str): The tuple in a string format.

    Returns:
    -------
    The tuple
    """
    return (1,) + tuple(map(int, value.strip("()").split(",")))
