import os

# Base directory for the project
base_dir = os.path.dirname(os.path.abspath(__file__))

plantuml_code_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "library.puml")
metamodel_image_path = os.path.join(base_dir, "llm_assistant", "gui_metamodel_spec", "gui_mm.png")
metamodel_text_path = os.path.join(base_dir, "llm_assistant", "gui_metamodel_spec", "gui_mm.html")
first_example_code_path = os.path.join(base_dir, "llm_assistant", "code", "gui_model_first_example.py")
second_example_code_path = os.path.join(base_dir, "llm_assistant", "code", "gui_model_second_example.py")

#python code for four pages of library web app:
first_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_main_page.py")
second_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_library_page.py")
third_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_book_page.py")
forth_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_author_page.py")

#python code for whole web app:
single_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_model", "gui_model_whole_pages.py")
