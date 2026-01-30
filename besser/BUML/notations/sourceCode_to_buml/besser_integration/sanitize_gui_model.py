import ast
import inspect
import os
import black  # pip install black
from besser.BUML.metamodel import gui as gui_metamodel


def get_allowed_args():
    """
    Collect allowed constructor arguments for each GUI metamodel class.
    Also track which arguments are required (no default).
    """
    allowed = {}
    required = {}

    for name, cls in gui_metamodel.__dict__.items():
        if isinstance(cls, type):
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:]  # skip "self"

            allowed[name] = [p.name for p in params]
            required[name] = [
                p.name for p in params if p.default is inspect._empty
            ]

    return allowed, required


class ArgSanitizer(ast.NodeTransformer):
    """
    Walks through the AST and removes invalid args for GUI components.
    Also adds missing required args with default "".
    """

    def __init__(self, allowed_args, required_args):
        super().__init__()
        self.allowed_args = allowed_args
        self.required_args = required_args

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            cls_name = node.func.id
            if cls_name in self.allowed_args:
                valid = set(self.allowed_args[cls_name])
                req = set(self.required_args[cls_name])

                # Keep only valid keyword arguments
                node.keywords = [kw for kw in node.keywords if kw.arg in valid]

                # Track which args are already provided
                provided = {kw.arg for kw in node.keywords if kw.arg is not None}

                # Add missing required arguments with default ""
                missing = req - provided
                for arg in missing:
                    node.keywords.append(
                        ast.keyword(arg=arg, value=ast.Constant(value=""))
                    )

        return self.generic_visit(node)


def sanitize_generated_gui_model(output_folder: str, output_file: str = None):
    """
    Clean the generated GUI model by stripping invalid constructor args
    and filling missing required args with default "".
    """

    gui_output_dir = os.path.join(output_folder, "gui_model")
    gui_output_file = os.path.join(gui_output_dir, "generated_gui_model.py")

    with open(gui_output_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    allowed_args, required_args = get_allowed_args()
    sanitized_tree = ArgSanitizer(allowed_args, required_args).visit(tree)

    # Convert AST back to code (Python 3.9+)
    new_code = ast.unparse(sanitized_tree)

    # Format with black → enforce one-liners by setting high line_length
    formatted_code = black.format_str(new_code, mode=black.Mode(line_length=500))

    output_file = gui_output_file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_code)

    print(f"[Sanitizer] Validated GUI model saved to {output_file}")
    return output_file
