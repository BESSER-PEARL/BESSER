import ast
import os
import shutil
from pathlib import Path
import copy
import glob
from collections import defaultdict

INPUT_DIR = "sample_models"
OUTPUT_DIR = "mutants"

file_summary = {}
global_summary = defaultdict(int)


class Mutator(ast.NodeTransformer):

    def __init__(self):
        self.mutants = []

    # M1 Arithmetic Operator Replacement
    def visit_BinOp(self, node):
        self.generic_visit(node)

        replacements = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
            ast.Div: ast.Mult
        }

        for op_type, new_op in replacements.items():
            if isinstance(node.op, op_type):
                mutant = copy.deepcopy(node)
                mutant.op = new_op()
                self.mutants.append((node, mutant, "M1"))

        return node

    # M2 Relational Operator Replacement
    # M5 Comparison Operator Replacement
    def visit_Compare(self, node):
        self.generic_visit(node)

        ror = {
            ast.Gt: ast.Lt,
            ast.Lt: ast.Gt,
            ast.GtE: ast.LtE,
            ast.LtE: ast.GtE
        }

        cor = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq
        }

        for op_type, new_op in ror.items():
            if isinstance(node.ops[0], op_type):
                mutant = copy.deepcopy(node)
                mutant.ops[0] = new_op()
                self.mutants.append((node, mutant, "M2"))

        for op_type, new_op in cor.items():
            if isinstance(node.ops[0], op_type):
                mutant = copy.deepcopy(node)
                mutant.ops[0] = new_op()
                self.mutants.append((node, mutant, "M5"))

        return node

    # M4 Incorrect return values
    def visit_Return(self, node):
        self.generic_visit(node)

        if node.value is not None:
            mutant = copy.deepcopy(node)
            mutant.value = ast.Constant(value=None)
            self.mutants.append((node, mutant, "M4"))

        return node

    # M3 Removing assignments + M7 Updating attribute
    def visit_Assign(self, node):
        self.generic_visit(node)

        # M3 remove assignment
        self.mutants.append((node, ast.Pass(), "M3"))

        # M7 update attribute name
        if isinstance(node.targets[0], ast.Name):
            mutant = copy.deepcopy(node)
            old_name = node.targets[0].id
            mutant.targets[0].id = old_name + "_mut"
            self.mutants.append((node, mutant, "M7"))

        return node

    # M8 Updating operation signature
    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        mutant = copy.deepcopy(node)

        for arg in mutant.args.args:
            arg.arg = arg.arg + "_mut"

        self.mutants.append((node, mutant, "M8"))

        return node


# M6 Removing attributes or operations
def remove_attr_or_method(tree):

    new_body = []

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.FunctionDef)):
            continue

        new_body.append(node)

    tree.body = new_body

    return tree, "M6"


def generate_mutants_for_file(filepath, output_dir):

    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    mutator = Mutator()
    mutator.visit(tree)

    removed_tree, desc = remove_attr_or_method(copy.deepcopy(tree))
    mutator.mutants.append((None, removed_tree, desc))

    mutant_counts = defaultdict(int)
    mutant_id = 0

    for original, mutant, mtype in mutator.mutants:

        if original is not None:

            mutant_tree = ast.parse(source)

            class ReplaceNode(ast.NodeTransformer):
                def visit(self, node):
                    if ast.dump(node) == ast.dump(original):
                        return mutant
                    return super().visit(node)

            mutant_tree = ReplaceNode().visit(mutant_tree)

        else:
            mutant_tree = mutant

        ast.fix_missing_locations(mutant_tree)

        mutant_code = f"# Mutation Operator: {mtype}\n"
        mutant_code += ast.unparse(mutant_tree)

        model_name = filepath.split("\\")[1].replace("output_","")

        mutant_filename = f"{model_name}_mutant_{mutant_id}.py"
        mutant_path = Path(output_dir) / mutant_filename

        with open(mutant_path, "w") as f:
            f.write(mutant_code)

        mutant_counts[mtype] += 1
        global_summary[mtype] += 1

        mutant_id += 1

    file_summary[model_name] = dict(mutant_counts)

    print("\nFile:", model_name)
    print("  Total mutants:", mutant_id)

    for k, v in sorted(mutant_counts.items()):
        print(f"  {k}: {v}")


def list_files_glob(pattern):
    return glob.glob(pattern)


def main():

    files = list_files_glob(INPUT_DIR + "/output_*")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)

    for file in files:
        generate_mutants_for_file(os.path.join(file, "classes.py"), OUTPUT_DIR)

    print("\n============================")
    print("FINAL SUMMARY")
    print("============================")

    for file, data in file_summary.items():

        total = sum(data.values())

        print("\nFile:", file)
        print("  Total mutants:", total)

        for k, v in sorted(data.items()):
            print(f"  {k}: {v}")

    print("\n============================")
    print("GLOBAL MUTATION SUMMARY")
    print("============================")

    total_all = sum(global_summary.values())
    print("Total mutants across all files:", total_all)

    for k, v in sorted(global_summary.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()