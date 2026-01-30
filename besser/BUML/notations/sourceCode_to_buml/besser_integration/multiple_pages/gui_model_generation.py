import os
import requests
import retrying
from besser.BUML.notations.sourceCode_to_buml.utilities import read_file_contents, encode_image
from besser.BUML.notations.sourceCode_to_buml.config import (
    first_gui_code_path,
    second_gui_code_path,
    third_gui_code_path,
    fourth_gui_code_path,
    single_gui_code_path,
    metamodel_image_path,
    metamodel_text_path, base_dir
)


@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_fix_string_property_references(api_key, python_code, structural_model_path):
    # Encode the images

    structural_model_contents = read_file_contents(structural_model_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    #revise_prompt = "Please consider the Python code provided below.\n"
    #revise_prompt += "Ensure using the actual Property objects (already defined in the structural model), not strings.\n"
    #revise_prompt += "After revising the Python code, please reply with the updated version.\n"y


    #revise_prompt = """
    #Please consider the Python code provided below. Your task is to revise the code with the following goals:

    #1. Ensure that all references to screen objects are **only used after they have been defined**.
    #2. Never use forward-references to screens that have not yet been declared.
    #3. Make sure that **Property** objects defined in the structural model are reused directly instead of redefining them or using strings.
    #4. If an element refers to another (like a `Button` linking to a screen), ensure that referenced screen is already declared **before** it's used.

    #Please return the revised Python code with the above corrections applied.
    #"""

    revise_prompt = """
    You are given a Python code snippet that defines a structural and GUI model for an application. Please revise the code with the following objectives:

    1. **Avoid forward references**: Ensure that no screen, button, or any GUI element refers to another object (like a `Screen`, `DataSourceElement`, etc.) that hasn't already been declared above in the code. All references must be to objects defined earlier in the code.

    2. **Sequential integrity**: Maintain a logical and sequential order of declarations. That is, first declare all screens and components, and only then establish interconnections (e.g., assigning a `targetScreen`).

    3. **Preserve and reuse structural elements**: For all `Property` objects and classes, make sure they are reused throughout the code instead of being recreated or replaced with string literals.

    4. **Late-binding of cross-references**: Where necessary, defer linking (like setting `targetScreen`) until all referenced objects have been defined. This can be done after the main block of screen and element declarations.

    Please return the revised Python code with the above changes applied. Maintain all functional and visual configurations as in the original code.
    """


    messages = [
        {
            "role": "system",
            "content": "As a Python developer, your task is to refine the given code by ensuring two things: (1) all references to screen objects occur only after those screens have been defined, and (2) actual Property objects from the structural model are used instead of string literals."

        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": revise_prompt
                },
                {
                    "type": "text",
                    "text": python_code
                },
            ]
        },
    ]

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

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120)

    response_json = response.json()

    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

    try:
        completed_code = response_json['choices'][0]['message']['content']
        return completed_code
    except (KeyError, IndexError) as e:
        print(f"Error occurred: {e}")
        return None




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


    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120)

    response_json = response.json()

    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

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
    improve_prompt += "Also, please specify the 'is_main_page=True' attribute for main page of the application.\n"
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

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
        )

    response_json = response.json()

    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

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
    revision_prompt += "Please specify the 'is_main_page=True' attribute for main page of the application.\n"
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
        with open(file_path, 'r', encoding='utf-8') as file:
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

    response = requests.post("" \
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=120
    )

    response_json = response.json()

    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

    try:
        improved_code = response_json['choices'][0]['message']['content']
        return improved_code
    except KeyError:
        return None


# Function to call GPT-4o with the direct prompt method
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4o_call_whole_app(api_key, base64_metamodel, prompt, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, fourth_gui_code_path, single_gui_code_path, metamodel_text_path, structural_model_path):

    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)
    metamodel_text_contents = read_file_contents(metamodel_text_path)

    first_example_code_contents = read_file_contents(first_gui_code_path)
    second_example_code_contents = read_file_contents(second_gui_code_path)
    third_example_code_contents = read_file_contents(third_gui_code_path)
    fourth_example_code_contents = read_file_contents(fourth_gui_code_path)
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
        with open(file_path, 'r', encoding='utf-8') as file:
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

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120)

    response_json = response.json()


    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

    try:
        python_code = response_json['choices'][0]['message']['content']
        return python_code
    except KeyError:
        return None



def direct_prompting_whole_app(api_key, metamodel_image_path, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, fourth_gui_code_path, single_gui_code_path, metamodel_text_path, structural_model_path):

    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# GUI Python Code Integration for Web App\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate Python code that integrates GUI elements from multiple pages of a web application into a single unified GUI model.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "The GUI Python code files you will receive represent different pages of a web application designed for performing CRUD operations on a structure of entities. Each Python code defines the GUI elements for a specific page of the application.\n"
    direct_prompt += "This structure is defined in the structural model through Python code, representing entities, their attributes, and their relationships.\n"
    direct_prompt += "Your goal is to integrate these individual GUI models into a cohesive Python code that accurately reflects the entire web application's UI design.\n"
    direct_prompt += "Ensure the integration maintains consistency in design, navigation, and functionality across the different pages.\n\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. You will be provided with multiple GUI models for different pages of a web application.\n"
    direct_prompt += "2. Each GUI model corresponds to a specific page in the web application.\n"
    direct_prompt += "3. You also will be provided with example GUI models for each page of the web application and its expected Python output for the whole of application.\n"
    direct_prompt += "4. Your task is to combine the individual GUI models into a single Python file that integrates **all pages**, **without modifying the content of the original Python code files**.\n"
    direct_prompt += "5. Ensure smooth navigation between pages and maintain a coherent user experience, using the proper navigation actions and links.\n"
    direct_prompt += "6. The integrated code must adhere to the GUI metamodel provided, and all UI elements (buttons, screens, actions) must follow the same structure and design principles.\n"
    direct_prompt += "7. The code should reflect the layout and styling specifications from each page's GUI model.\n"
    direct_prompt += "8. Consider grouping related screens under one module, and ensure that 'screens' is an attribute containing all the screens in the final app.\n"
    direct_prompt += "9. Make sure the final model retains the design consistency across all screens and components.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Integrate the provided GUI models into a single Python code (**with considerig the content of all python code files**) that reflects the complete UI design of the web application. This should include handling multiple screens, actions, and their proper layout and navigation.\n"



    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)

    # Call gpt4o_call to generate the Python code using the direct prompt method
    python_code = gpt4o_call_whole_app(api_key, base64_metamodel, direct_prompt, gui_models_paths, first_gui_code_path, second_gui_code_path, third_gui_code_path, fourth_gui_code_path, single_gui_code_path, metamodel_text_path, structural_model_path)


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
    python_code = direct_prompting_whole_app(api_key, metamodel_image_path, gui_models_paths,
                                             first_gui_code_path, second_gui_code_path,
                                             third_gui_code_path, fourth_gui_code_path,
                                             single_gui_code_path, metamodel_text_path,
                                             structural_model_path)

    # Save the generated code to a file
    #if python_code:
        # Specify the desired output file name
        #output_file_name = os.path.join(gui_output_dir, "generated_gui_model_1.py")
        #with open(output_file_name, "w", encoding="utf-8") as file:
            #file.write(python_code)
        #print(f"Generated Python code saved to {output_file_name}")
    #else:
        #print("Failed to generate Python code.")


    # Generate the revised Python code using the self-improvement method"
    improved_code = gpt4_self_improvement_whole_app(api_key, metamodel_image_path, gui_models_paths,
                                                    metamodel_text_path, python_code, structural_model_path)

    # Save the generated code to a file
    #if improved_code:
        # Specify the desired output file name
        #output_file_name = os.path.join(gui_output_dir, "generated_gui_model_2.py")
        #with open(output_file_name, "w", encoding="utf-8") as file:
            #file.write(improved_code)
        #print(f"Generated revise code saved to {output_file_name}")
    #else:
        #print("Failed to generate revise code.")


    refactored_code = gpt4_refactor(api_key, improved_code)
    # Save the generated code to a file
    #if refactored_code:
        # Specify the desired output file name
        #output_file_name = os.path.join(gui_output_dir, "generated_gui_model_3.py")
        #with open(output_file_name, "w", encoding="utf-8") as file:
            #file.write(refactored_code)
        #print(f"Generated refactor code saved to {output_file_name}")
    #else:
        #print("Failed to generate revise code.")


    completed_code = gpt4_add_navigation_between_screens(api_key, refactored_code,
                                                         navigation_image_path,
                                                         pages_order_file_path)
    # Save the generated code to a file
    #if completed_code:
        # Specify the desired output file name
        #output_file_name = os.path.join(gui_output_dir, "generated_gui_model_4.py")
        #with open(output_file_name, "w", encoding="utf-8") as file:
            #file.write(completed_code)
        #print(f"Generated complete code saved to {output_file_name}")
    #else:
        #print("Failed to generate revise code.")

    final_code = gpt4_fix_string_property_references (api_key, completed_code, structural_model_path)
    # Save the generated code to a file
    if final_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py")
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(final_code)
        print(f"Generated integrated GUI model saved to {output_file_name}")
    else:
        print("Failed to generate revise code.")


