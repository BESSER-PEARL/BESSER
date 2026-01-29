import os
import requests
import retrying
from besser.BUML.notations.mockup_to_structural.utilities import *
from besser.BUML.notations.mockup_to_structural.utilities import *
from besser.BUML.notations.mockup_to_buml.config import *


@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_add_navigation_between_screens(api_key, python_code, navigation_image_path, pages_order_file_path):
    # Encode the images
    if navigation_image_path:
       base64_navigation = encode_image(navigation_image_path)

    if pages_order_file_path:
       pages_order_file_path = read_file_contents(pages_order_file_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    complete_prompt = "Please review the Python code below.\n"
    complete_prompt += "You will receive an text file that specifies the order of pages.\n"
    complete_prompt += "Please reorder the content based on the order specified in a text file, but in the reverse order compared to the provided sequence.\n"
    complete_prompt += "You will receive an image illustrating the navigation flow between multiple screens of the web application.\n"
    complete_prompt += "Complete the code by just specifying target screens for buttons with 'actionType=ButtonActionType.Navigate'.\n"
    complete_prompt += "Focus solely on adding the target screen attribute for buttons with 'actionType=ButtonActionType.Navigate'.\n"
    complete_prompt += "Once you've made the necessary additions, kindly respond with the updated version of the Python code.\n"

    messages = [
        {
            "role": "system",
            "content": "As a developer, your task is to complete the given Python code to add target screen features to the buttons with 'actionType=ButtonActionType.Navigate'."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": complete_prompt
                },
                {
                    "type": "text",
                    "text": python_code
                },
            ]
        },
    ]

    if navigation_image_path:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the navigation image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_navigation}",
                            "detail": "high"
                        }
                    }
                ]
            }
        )
    if pages_order_file_path:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "To view the order of pages, refer to the following file: [expected output]\n{pages_order_file_path}"
                    }
                ]
            }
        )
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_json = response.json()

    try:
        completed_code = response_json['choices'][0]['message']['content']
        return completed_code
    except (KeyError, IndexError) as e:
        print(f"Error occurred: {e}")
        return None


@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_refactor(api_key, python_code):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    improve_prompt = "Please consider the Python code provided below.\n"
    improve_prompt += "Ensure that any buttons labeled with 'actionType=ButtonActionType.Navigate' without a specified target screen and starting with 'back' are removed.\n"
    improve_prompt += "Additionally, remove these buttons from the 'view_elements' of each screen where they are present.\n"
    improve_prompt += "After revising the Python code, please reply with the updated version.\n"

    messages = [
        {
            "role": "system",
            "content": "As a developer, your task is to refine the given Python code to align with Python syntax standards. Focus on eliminating any non-conforming elements and enhancing code clarity."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": improve_prompt
                },
                {
                    "type": "text",
                    "text": python_code
                },
            ]
        },
    ]

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()

    try:
        refactored_code = response_json['choices'][0]['message']['content']
        return refactored_code
    except (KeyError, IndexError) as e:
        print(f"Error occurred: {e}")
        return None

# Function to facilitate self-improvement of a Python code representing GUI elements using the GPT-4o model
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement_whole_app(api_key, base64_metamodel, gui_models_paths, metamodel_text_path, python_code, structural_model_path):

    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)

    metamodel_text_contents = read_file_contents(metamodel_text_path)
    structural_model_contents = read_file_contents(structural_model_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    ## revision_prompt
    revision_prompt = ""
    revision_prompt += "As an expert Web GUI developer, I need your assistance in improving a Python code that that represents the integration of GUI elements from multiple pages of a web application."
    revision_prompt += "The given Python code follows the GUI metamodel but might be missing or contain incorrect elements compared to the GUI models (python files).\n"
    revision_prompt += "I have provided you with the GUI metamodel image file and a description file to assist you in the revision process.\n"
    revision_prompt += "The Python code you receive as input is related to whole of the pages of an application designed for performing database operations (CRUD) on a structure of entities.\n"
    revision_prompt += "This structure is defined in the structural model by Python code, representing the entities, their attributes, and the relationships between them.\n"
    revision_prompt += "Please note that in this page, all attributes or a subset of the attributes of the desired class may be considered. You can refer to the list of attributes of the class according to the structural model and incorporate these attributes in the generation of Python code.\n\n"
    revision_prompt += "Your task is to carefully compare the GUI elements in the GUI models with the ones in the Python code and make the necessary revisions.\n"
    revision_prompt += "Please remove any parts of the code that are related to the metaclasses definition of metamodel.\n"
    revision_prompt += "Please remove any hading or formating lines that is not true according the syntax of python code in the Python code.\n"
    revision_prompt += "Once you have revised the Python code, please respond with the updated version.\n"

    messages = [
        {
            "role": "system",
            "content": "You are a developer. Improve the given Python code to ensure that it includes all the GUI elements from the mock-up image."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": revision_prompt
                },
                {
                    "type": "text",
                    "text": python_code
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_metamodel}",
                        "detail": "high"
                    }
                }
            ]
        },
    ]
    for file_path in gui_models_paths:
        with open(file_path, 'r') as file:
            file_content = file.read()
            file_name = os.path.basename(file_path)

            messages.append(
                {
                    "role": "user",
                    "content": [
                       {
                         "type": "text",
                        "text": f"Here is the content of the file {file_name}:"
                       },
                       {
                        "type": "text",
                        "text": file_content
                       }
                    ]
                }
            )

    if metamodel_text_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's an explanation of the GUI metamodel:"
                    },
                    {
                        "type": "text",
                        "text": metamodel_text_contents
                    },
                ]
            }
        )
    if structural_model_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the provided structural model:"
                    },
                    {
                        "type": "text",
                        "text": structural_model_contents
                    },
                ]
            }
        )

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()

    try:
        improved_code = response_json['choices'][0]['message']['content']
        return improved_code
    except KeyError:
        return None


# Function to call GPT-4o with the direct prompt method
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4o_call_whole_app(api_key, base64_metamodel, prompt, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, forth_gui_code_path, single_gui_code_path, metamodel_text_path, structural_model_path):

    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)
    metamodel_text_contents = read_file_contents(metamodel_text_path)

    first_example_code_contents = read_file_contents(first_gui_code_path)
    second_example_code_contents = read_file_contents(second_gui_code_path)
    third_example_code_contents = read_file_contents(third_gui_code_path)
    fourth_example_code_contents = read_file_contents(forth_gui_code_path)
    single_example_code_contents = read_file_contents(single_gui_code_path)

    structural_model_contents = read_file_contents(structural_model_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a developer. Generate Python code that integrates the provided GUI models into a single Python code representation that reflects the complete UI design of the web application."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_metamodel}",
                        "detail": "high"
                    }
                }
            ]
        },
    ]
    for file_path in gui_models_paths:
        with open(file_path, 'r') as file:
            file_content = file.read()
            file_name = os.path.basename(file_path)

            messages.append(
                {
                    "role": "user",
                    "content": [
                       {
                         "type": "text",
                        "text": f"Here is the content of the file {file_name}:"
                       },
                       {
                        "type": "text",
                        "text": file_content
                       }
                    ]
                }
            )
    if first_gui_code_path:
        messages.append(
            {
               "role": "assistant",
               "content": [
                    {
                        "type": "text",
                        "text": "Here's the example of the GUI models for each page of the web application and the expected Python output for the whole application:"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the Python code for the main page of the app: [First Page]\n{first_example_code_contents}"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the Python code for the second page of the app: [Second Page]\n{second_example_code_contents}"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the Python code for the third page of the app: [Third Page]\n{third_example_code_contents}"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the Python code for the fourth page of the app: [Fourth Page]\n{fourth_example_code_contents}"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the Python code for all pages of the app: [expected output]\n{single_example_code_contents}"
                    },
            ]
        }
    )

    if metamodel_text_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's an explanation of the GUI metamodel:"
                    },
                    {
                        "type": "text",
                        "text": metamodel_text_contents
                    },
                ]
            }
        )
    if structural_model_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the provided structural model:"
                    },
                    {
                        "type": "text",
                        "text": structural_model_contents
                    },
                ]
            }
        )

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()

    try:
        python_code = response_json['choices'][0]['message']['content']
        return python_code
    except KeyError:
        return None

# Function to send a direct prompt to GPT-4o for generating Python code
def direct_prompting_whole_app(api_key, metamodel_image_path, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, forth_gui_code_path,  single_gui_code_path, metamodel_text_path, structural_model_path):


    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# GUI Python Code Integration for Web App\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate Python code that represents the integration of GUI elements from multiple pages of a web application into a single GUI model.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "The GUI python code files you will receive are related to different pages of a web application designed for performing database operations (CRUD) on a structure of entities. Each Python code represents a specific page's GUI elements.\n"
    direct_prompt += "This structure is defined in the structural model by Python code, representing the entities, their attributes, and the relationships between them.\n"
    direct_prompt += "Your goal is to integrate these individual GUI models into a cohesive Python code that represents the entire web application's UI design elements.\n"
    direct_prompt += "Ensure that the integration maintains consistency in design and functionality across the different pages.\n\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. You will be provided with example GUI models for each page of the web application and its expected Python output for the whole of application.\n"
    direct_prompt += "2. Your task is to combine these individual GUI models into a single Python code representation that reflects the complete UI design of the application.\n"
    direct_prompt += "3. The integrated code should ensure smooth navigation between the different pages and maintain a coherent user experience.\n"
    direct_prompt += "4. The Python code should adhere to the GUI metamodel provided for the web application.\n"
    direct_prompt += "5. Make sure to follow the design principles and layout specifications of each page while integrating them into the final GUI model.\n\n"
    direct_prompt += "6. Consider one module for the app that has a 'screens' attribute including all screens of the app, similar to the style of the provided example.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Integrate the provided GUI models into a single Python code representation that reflects the complete UI design of the web application.\n"


    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)

    # Call gpt4o_call to generate the Python code using the direct prompt method
    python_code = gpt4o_call_whole_app(api_key, base64_metamodel, direct_prompt, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, forth_gui_code_path, single_gui_code_path, metamodel_text_path, structural_model_path)

    return python_code

def run_pipeline_gui_model_generation(api_key: str, navigation_image_path: str, pages_order_file_path: str, output_folder: str):

    structural_dir = os.path.join(output_folder, "buml")
    structural_model_path = os.path.join(structural_dir, "model.py")

    folder_path = os.path.join(base_dir, "output", "gui_models")

    # Ensure the gui_models directory exists
    os.makedirs(folder_path, exist_ok=True)


    gui_output_dir=os.path.join(output_folder, "gui_model")
    os.makedirs(gui_output_dir, exist_ok=True)

    # List all files in the folder
    gui_models_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.py')]


    # Generate the Python code using the direct prompt method
    python_code = direct_prompting_whole_app(api_key, metamodel_image_path, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, forth_gui_code_path, single_gui_code_path, metamodel_text_path, structural_model_path)

    # Save the generated code to a file
    if python_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py") # Specify the desired output file name
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(python_code)
    else:
        print("Failed to generate Python code.")


    # Generate the revised Python code using the self-improvement method"
    improved_code = gpt4_self_improvement_whole_app(api_key, metamodel_image_path, gui_models_paths, metamodel_text_path, python_code, structural_model_path)
    # Save the generated code to a file
    if improved_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py") # Specify the desired output file name
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(improved_code)
    else:
        print("Failed to generate revise code.")


    refactored_code = gpt4_refactor(api_key, improved_code)
    # Save the generated code to a file
    if refactored_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py") # Specify the desired output file name
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(refactored_code)
    else:
        print("Failed to generate revise code.")


    completed_code = gpt4_add_navigation_between_screens(api_key, refactored_code, navigation_image_path, pages_order_file_path)
    # Save the generated code to a file
    if completed_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py") # Specify the desired output file name
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(completed_code)
        print(f"Generated Python code saved to {output_file_name}")
    else:
        print("Failed to generate revise code.")
