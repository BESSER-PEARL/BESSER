import ast
import os
import shutil
from pathlib import Path
import copy
import glob

INPUT_DIR = "sample_models"
OUTPUT_DIR = "mutants"

class Mutator(ast.NodeTransformer):

    def __init__(self):
        self.mutants = []

    def visit_BinOp(self, node):
        self.generic_visit(node)
        # Arithmetic Operator Replacement (AOR)
        replacements = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
        for op_type, new_op in replacements.items():
            if isinstance(node.op, op_type):
                mutant = copy.deepcopy(node)
                mutant.op = new_op()
                self.mutants.append((node, mutant, "AOR: Arithmetic Operator Replacement"))
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        # Relational Operator Replacement (ROR)
        rel_replacements = {ast.Gt: ast.Lt, ast.Lt: ast.Gt, ast.GtE: ast.LtE, ast.LtE: ast.GtE}
        # Comparison Operator Replacement (COR)
        comp_replacements = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}

        for op_type, new_op in {**rel_replacements, **comp_replacements}.items():
            if isinstance(node.ops[0], op_type):
                mutant = copy.deepcopy(node)
                mutant.ops[0] = new_op()
                desc = "ROR" if op_type in rel_replacements else "COR"
                self.mutants.append((node, mutant, f"{desc}: Operator Replacement"))
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        # Incorrect return values
        if node.value is not None:
            mutant = copy.deepcopy(node)
            mutant.value = ast.Constant(value=None)
            self.mutants.append((node, mutant, "Incorrect Return Value"))
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        # Removing assignments
        self.mutants.append((node, ast.Pass(), "Removed Assignment"))
        # Updating attribute name
        if isinstance(node.targets[0], ast.Name):
            mutant = copy.deepcopy(node)
            old_name = node.targets[0].id
            mutant.targets[0].id = old_name + "_mutant"
            self.mutants.append((node, mutant, f"Updated Attribute Name: {old_name} -> {mutant.targets[0].id}"))
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        # Updating signature of operation
        mutant = copy.deepcopy(node)
        for i, arg in enumerate(mutant.args.args):
            arg.arg = arg.arg + "_m"
        self.mutants.append((node, mutant, f"Updated Operation Signature: {node.name}"))
        return node

def remove_attr_or_method(tree):
    """Remove attributes or operations at module level"""
    new_body = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.FunctionDef)):
            continue  # remove
        new_body.append(node)
    tree.body = new_body
    return tree, "Removed Attributes or Operations"

def generate_mutants_for_file(filepath, output_dir):
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    mutator = Mutator()
    mutator.visit(tree)

    # Module level removal of attributes/methods
    removed_tree, removed_desc = remove_attr_or_method(copy.deepcopy(tree))
    mutator.mutants.append((None, removed_tree, removed_desc))

    mutant_id = 0
    for original, mutant, desc in mutator.mutants:
        mutant_tree = ast.parse(source) if original is not None else mutant
        if original is not None:
            class ReplaceNode(ast.NodeTransformer):
                def visit(self, node):
                    if ast.dump(node) == ast.dump(original):
                        return mutant
                    return super().visit(node)
            mutant_tree = ReplaceNode().visit(mutant_tree)

        ast.fix_missing_locations(mutant_tree)
        mutant_code = f"# Mutant Operator: {desc}\n" + ast.unparse(mutant_tree)

        mutant_filename = f"{filepath.split("\\")[1].split("_")[1]}_mutant_{mutant_id}.py"
        mutant_path = Path(output_dir) / mutant_filename
        with open(mutant_path, "w") as f:
            f.write(mutant_code)

        mutant_id += 1

def list_files_glob(pattern='', recursive=False):
    return glob.glob(pattern, recursive=recursive)

def main():
    files = list_files_glob(INPUT_DIR + "/output_*")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    for file in files:
        generate_mutants_for_file(os.path.join(file, "classes.py"), OUTPUT_DIR)

if __name__ == "__main__":
    main()