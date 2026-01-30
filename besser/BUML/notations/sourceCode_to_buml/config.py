import os

# Base directory for the project
base_dir = os.path.dirname(os.path.abspath(__file__))

### Paths for GUI model conversion ###
metamodel_image_path = os.path.join(base_dir, "llm_assistant", "gui_metamodel_spec", "gui_mm.png")
metamodel_text_path = os.path.join(base_dir, "llm_assistant", "gui_metamodel_spec", "gui_mm.html")
# Example code paths:
first_example_source_code_path = os.path.join(base_dir, "llm_assistant", "example", "source_code", "home.html")
second_example_source_code_path = os.path.join(base_dir, "llm_assistant", "example", "source_code", "library_list.html")
first_example_gui_code_path = os.path.join(base_dir, "llm_assistant", "example", "gui_models", "gui_model_first_example.py")
second_example_gui_code_path = os.path.join(base_dir, "llm_assistant", "example", "gui_models", "gui_model_second_example.py")

# Example code paths for SVG phase:
gui_code_example_path = os.path.join(base_dir, "llm_assistant", "example", "svg", "gui_model_book_page.py")
svg_code_example_path = os.path.join(base_dir, "llm_assistant", "example", "svg", "BookListScreen.svg")

#python code for four pages of library web app:
first_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_main_page.py")
second_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_library_page.py")
third_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_book_page.py")
fourth_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_models", "gui_model_author_page.py")


#example python code for whole web app:
single_gui_code_path = os.path.join(base_dir, "llm_assistant", "gui_model", "gui_model_whole_pages.py")





