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

### Paths for GUI model conversion ###
metamodel_image_path = os.path.join(base_dir, "images", "gui_mm.png")
metamodel_text_path = os.path.join(base_dir, "files", "gui_mm.html")
# Example code paths:
first_example_code_path = os.path.join(base_dir, "example", "code", "gui_model_first_example.py")
second_example_code_path = os.path.join(base_dir, "example", "code", "gui_model_second_example.py")
# Path to structural model:
structural_model_path = os.path.join(base_dir, "output", "buml", "model.py")
# Path to the folder containing image mockups:
mockup_folder_path = os.path.join(base_dir, "mockup_image")


#### Paths for handling multiple pages ###
folder_path = os.path.join(base_dir, "output", "gui_models")
gui_output_dir=os.path.join(base_dir, "output", "gui_model")
gui_output_file=os.path.join(base_dir, "output", "gui_model", "generated_gui_model.py")


#python code for four pages of library web app:
first_gui_code_path = os.path.join(base_dir, "files", "models", "gui_model_main_page.py")
second_gui_code_path = os.path.join(base_dir, "files", "models", "gui_model_library_page.py")
third_gui_code_path = os.path.join(base_dir, "files", "models", "gui_model_book_page.py")
forth_gui_code_path = os.path.join(base_dir, "files", "models", "gui_model_author_page.py")

#python code for whole web app:
single_gui_code_path = os.path.join(base_dir, "files", "example_gui_model", "gui_model_whole_pages.py")

