import os

# Base directory for the project
base_dir = os.path.dirname(os.path.abspath(__file__))


### Paths for PlantUML and code generation ###
first_code_file_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "library_list.html")
second_code_file_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "multiple_pages", "author_list.html")
third_code_file_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "multiple_pages", "book_list.html")
fourth_code_file_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "multiple_pages", "home.html")

plantuml_code_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "library.puml")
plantuml_multiple_pages_code_example_path = os.path.join(base_dir, "llm_assistant", "plantuml", "multiple_pages", "library.puml")



