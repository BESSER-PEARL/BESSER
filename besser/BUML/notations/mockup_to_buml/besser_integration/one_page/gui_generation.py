import os
import requests
import retrying
from besser.BUML.notations.mockup_to_structural.utilities import *
from besser.BUML.notations.mockup_to_structural.utilities import *
from besser.BUML.notations.mockup_to_buml.config import *


# Function to facilitate self-improvement of a Python code representing GUI elements using the GPT-4o model
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement(api_key, mockup_image_path, base64_metamodel, base64_mockup, metamodel_text_path, python_code, structural_model_path):

    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)
    base64_mockup = encode_image(mockup_image_path)
    metamodel_text_contents = read_file_contents(metamodel_text_path)
    structural_model_contents = read_file_contents(structural_model_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    ## revision_prompt
    revision_prompt = ""
    revision_prompt += "As an expert Web GUI developer, I need your assistance in improving a Python code that represents the GUI elements of a UI mock-up.\n"
    revision_prompt += "The given Python code follows the GUI metamodel but might be missing or contain incorrect elements compared to the GUI mock-up image.\n"
    revision_prompt += "I have provided you with the GUI metamodel image file and a description file to assist you in the revision process.\n"
    revision_prompt += "The Python code you receive as input is related to one of the pages of an application designed for performing database operations (CRUD) on a structure of entities.\n"
    revision_prompt += "This structure is defined in the structural model by Python code, representing the entities, their attributes, and the relationships between them.\n"
    revision_prompt += "Please note that in this page, all attributes or a subset of the attributes of the desired class may be considered. You can refer to the list of attributes of the class according to the structural model and incorporate these attributes in the generation of Python code.\n\n"
    revision_prompt += "Your task is to carefully compare the GUI elements in the mock-up image with the ones in the Python code and make the necessary revisions.\n"
    revision_prompt += "Please remove any parts of the code that are related to the metaclasses definition of metamodel.\n"
    revision_prompt += "Once you have revised the Python code, please respond with the updated version.\n"
    revision_prompt += "In the end of the python code that you generated, please mention which class this page is related to in the structural model. Additionally, provide a list of attributes of this class based on the structural model."

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
                        "url": f"data:image/jpeg;base64,{base64_mockup}",
                        "detail": "high"
                    }
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
def gpt4o_call(api_key, mockup_image_path, base64_metamodel, base64_mockup, prompt, first_example_code_path, second_example_code_path, metamodel_text_path, structural_model_path):

    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)
    base64_mockup = encode_image(mockup_image_path)
    metamodel_text_contents = read_file_contents(metamodel_text_path)
    first_example_code_contents = read_file_contents(first_example_code_path)
    second_example_code_contents = read_file_contents(second_example_code_path)
    structural_model_contents = read_file_contents(structural_model_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a developer. Generate Python code that represents the GUI elements in the provided mock-up while conforming to the GUI metamodel."
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
                        "url": f"data:image/jpeg;base64,{base64_mockup}",
                        "detail": "high"
                    }
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

    if first_example_code_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the first example of a UI mock-up and the expected Python output:"
                    },
                    {
                        "type": "text",
                        "text": "You can view the example image at this link: [Example Image](https://imgurl.ir/uploads/p876089_main_page.png)"
                    },
                    {
                        "type": "text",
                        "text": first_example_code_contents
                    },
                ]
            }
        )
    if second_example_code_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the second example of a UI mock-up and the expected Python output:"
                    },
                    {
                        "type": "text",
                        "text": "You can view the example image at this link: [Example Image](https://imgurl.ir/uploads/f832119_library_directory.png)"
                    },
                    {
                        "type": "text",
                        "text": second_example_code_contents
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
def direct_prompting(api_key, metamodel_image_path, mockup_image_path, first_example_code_path, second_example_code_path, metamodel_text_path, structural_model_path):

    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# UI Mock-up to Python Code Generation\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate Python code that represents the GUI elements of a given UI mock-up and conforms to the GUI metamodel.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "The UI Mock-up model you receive as input is related to one of the pages of a web application designed for performing database operations (CRUD) on a structure of entities.\n"
    direct_prompt += "This structure is defined in the structural model by Python code, representing the entities, their attributes, and the relationships between them.\n"
    direct_prompt += "This mock-up is related to a page of the app that corresponds to one of the classes in the structural model.\n"
    direct_prompt += "Please note that in this page, all attributes or a subset of the attributes of the desired class may be considered. You can refer to the list of attributes of the class according to the structural model and incorporate these attributes in the generation of Python code.\n\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. I will provide you with example UI mock-ups and their expected Python output.\n"
    direct_prompt += "2. I will also provide you with a UI mock-up image as input.\n"
    direct_prompt += "3. I will also provide you with an image of the GUI metamodel and a description file for it.\n"
    direct_prompt += "4. Your goal is to generate Python code that accurately represents all GUI elements in the mock-up.\n"
    direct_prompt += "5. Please provide the Python code representing the GUI elements only, without any introductory or formatting lines.\n"
    direct_prompt += "6. It is crucial: the structure of the Python code you generate should be similar to the code samples provided as examples, while the content should be derived from the UI mockups you receive.\n"
    direct_prompt += "7. The generated code should adhere to the GUI metamodel.\n"
    direct_prompt += "8. Incorporate the provided examples and ensure the generated code aligns with the user's expectations.\n"
    direct_prompt += "9. Once you're ready, I will present you with a UI mock-up image, and you need to generate Python code for its GUI elements.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Respond with the Python code that represents the GUI elements of the provided UI mock-up.\n"

    # Encode the images
    base64_metamodel = encode_image(metamodel_image_path)
    base64_mockup = encode_image(mockup_image_path)

    # Call gpt4o_call to generate the Python code using the direct prompt method
    python_code = gpt4o_call(api_key, mockup_image_path, base64_metamodel, base64_mockup, direct_prompt, first_example_code_path,
    second_example_code_path, metamodel_text_path, structural_model_path)


    return python_code


def run_pipeline_gui_generation(api_key: str, mockup_image_path: str, output_folder: str):

    structural_dir = os.path.join(output_folder, "buml")
    structural_model_path = os.path.join(structural_dir, "model.py")

    # Generate the Python code using the direct prompt method
    python_code = direct_prompting(
        api_key,
        metamodel_image_path,
        mockup_image_path,
        first_example_code_path,
        second_example_code_path,
        metamodel_text_path,
        structural_model_path
    )

    # Define the gui subfolder inside the output folder
    gui_output_dir = os.path.join(output_folder, "gui_model")

    # Ensure the gui_model directory exists
    os.makedirs(gui_output_dir, exist_ok=True)


    # Save the generated code to a file
    if python_code:
        output_file_name = os.path.join(
            gui_output_dir, f"{get_file_name(mockup_image_path)}.py"
        )
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(python_code)
    else:
        print("Failed to generate Python code.")

    # Generate the revised Python code using the self-improvement method
    improved_code = gpt4_self_improvement(api_key, mockup_image_path, metamodel_image_path, mockup_image_path, metamodel_text_path, python_code, structural_model_path)


    # Save the generated revised code to a file
    if improved_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py")
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(improved_code)
        print(f"Generated Python code saved to {output_file_name}")
    else:
        print("Failed to generate revised code.")
