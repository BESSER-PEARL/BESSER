import os
from besser.BUML.notations.sourceCode_to_buml.utilities import get_file_name
from besser.BUML.notations.sourceCode_to_buml.besser_integration.one_page import *



def run_pipeline_gui_models_generation(api_key: str, source_code_folder_path: str,
                                       styling_file_path:str, output_folder: str) :

    structural_dir = os.path.join(output_folder, "buml")
    structural_model_path = os.path.join(structural_dir, "model.py")

    # List all files in the folder
    source_code_paths = [os.path.join(source_code_folder_path, f) for f in os.listdir(source_code_folder_path) if
                         os.path.isfile(os.path.join(source_code_folder_path, f))]

    # Define the gui_models subfolder inside the output folder
    folder_path = os.path.join(output_folder, "gui_models")

    # Ensure the gui_models directory exists
    os.makedirs(folder_path, exist_ok=True)

    for index, source_code_path in enumerate(source_code_paths, start=1):
        # Generate the Python code using the direct prompt method
        python_code = direct_prompting(api_key, metamodel_image_path, source_code_path,
                                       first_example_source_code_path, second_example_source_code_path,
                                       first_example_gui_code_path, second_example_gui_code_path,metamodel_text_path,
                                       structural_model_path)

        # Save the generated code to a file
        if python_code:
            # Specify the desired output file name
            output_file_name = os.path.join(folder_path, f"{get_file_name(source_code_path)}.py")
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(python_code)
            #print(f"Generated Python code saved to {output_file_name}")
        else:
            print("Failed to generate Python code.")


        # Generate the revised Python code using the self-improvement method"
        improved_code = gpt4_self_improvement(api_key, source_code_path, metamodel_image_path,
                                              metamodel_text_path, python_code, structural_model_path)


        # Save the generated code to a file
        if improved_code:
            # Specify the desired output file name
            output_file_name = os.path.join(folder_path, f"{get_file_name(source_code_path)}.py")
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(improved_code)
            #print(f"Generated revise code saved to {output_file_name}")
        else:
            print("Failed to generate revise code.")

        final_code = gpt4_fix_string_properties (api_key, improved_code, structural_model_path)

        # Save the generated code to a file
        if final_code:
            # Specify the desired output file name
            #print(f"Generated code saved to {output_file_name}")
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(final_code)
            print(f"Generated GUI model saved to {output_file_name}")
        else:
            print("Failed to generate revise code.")


        if styling_file_path:
            styling_code = gpt4_styling_prompting(api_key, source_code_path,
                                                  metamodel_image_path, metamodel_text_path,
                                                  final_code, styling_file_path)

            if styling_code:
                # Specify the desired output file name
                output_file_name = os.path.join(folder_path, f"{get_file_name(source_code_path)}.py")
                with open(output_file_name, "w", encoding="utf-8") as file:
                    file.write(styling_code)
                #print(f"Generated GUI model saved to {output_file_name}")
            else:
                print("Failed to generate revise code.")

            improved_styling_code = gpt4_self_improvement_styling_part(api_key, source_code_path,
                                                                       metamodel_image_path,
                                                                       metamodel_text_path,
                                                                       styling_code, styling_file_path)

            # Save the generated revised code to a file
            if improved_styling_code:
                output_file_name = os.path.join(folder_path, f"{get_file_name(source_code_path)}.py")
                with open(output_file_name, "w", encoding="utf-8") as file:
                    file.write(improved_styling_code)
                print(f"Generated GUI model saved to {output_file_name}")
            else:
                print("Failed to generate revise code.")
