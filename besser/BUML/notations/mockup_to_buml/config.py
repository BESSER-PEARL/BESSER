import os


# OpenAI API Key
api_key = ""

# Base directory for the project
base_dir = os.path.dirname(os.path.abspath(__file__))


# GUI images for four pages of library web app
first_gui_model_path = os.path.join(base_dir, "images", "main_page.png")
second_gui_model_path = os.path.join(base_dir, "images", "library_directory.png")
third_gui_model_path = os.path.join(base_dir, "images", "author_list.png")
forth_gui_model_path = os.path.join(base_dir, "images", "book_list.png")

### Paths for PlantUML and code generation ###
plantuml_code_example_path = os.path.join(base_dir, "example", "plantuml", "library.puml")
output_file_name = os.path.join(base_dir, "output", "plantuml", "generated_plantuml.puml")  # Specify the desired output file name
code_file = os.path.join(base_dir, "output", "plantuml", "generated_plantuml.puml")
output_dir = os.path.join(base_dir, "output", "plantuml")


# Path to structural model:
structural_model_path = os.path.join(base_dir, "output", "buml", "model.py")
# Path to the folder containing image mockups:
mockup_folder_path = os.path.join(base_dir, "mockup_image")


