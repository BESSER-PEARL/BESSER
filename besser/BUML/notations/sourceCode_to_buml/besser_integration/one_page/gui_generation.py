import os
import requests
import retrying
from besser.BUML.notations.sourceCode_to_buml.utilities import *
from besser.BUML.notations.sourceCode_to_buml.config import *

# Function to fix string properties in a Python code representing GUI elements using an LLM model
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_fix_string_properties(api_key, python_code, structural_model_path):

    structural_model_contents = read_file_contents(structural_model_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    revise_prompt = "Please consider the Python code provided below.\n"
    revise_prompt += "Ensure using the actual Property objects (already defined in the structural model), not strings.\n"
    revise_prompt += "After revising the Python code, please reply with the updated version.\n"

    messages = [
        {
            "role": "system",
            "content": "As a developer, your task is to refine the given Python code to "
            "ensure that it uses the actual Property objects (already defined in the structural model), "
            "rather than strings."


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
        timeout=120
        )

    response_json = response.json()


    try:
        completed_code = response_json['choices'][0]['message']['content']
        return completed_code
    except (KeyError, IndexError) as e:
        print(f"Error occurred: {e}")
        return None


# Function to facilitate self-improvement styling part of a Python code representing GUI elements using an LLM model
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement_styling_part(api_key, code_file_path, base64_metamodel, metamodel_text_path, python_code, styling_file_path):

    # Encode the image
    base64_metamodel = encode_image(metamodel_image_path)

    # Read file
    code_file_content = read_file_contents(code_file_path)
    metamodel_text_content = read_file_contents(metamodel_text_path)
    styling_file_content = read_file_contents(styling_file_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    ## revision_prompt
    styling_revision_prompt = ""
    styling_revision_prompt += "As an expert Web GUI developer, your task is to enhance the layout and styling aspects of the provided Python GUI code.\n"
    styling_revision_prompt += "The given Python code follows the GUI metamodel, but may be missing key styling or layout elements, or may contain incorrect styles compared to the source code and associated CSS styling files.\n"
    styling_revision_prompt += "To assist in your revision process, I have provided the GUI metamodel image file and a description file for reference.\n"
    styling_revision_prompt += "## Your Task:\n"
    styling_revision_prompt += "Carefully analyze the provided Python GUI code and compare it with the source code and CSS styling file to ensure the correct application of styling and layout elements (e.g., color, size, position, etc.).\n"
    styling_revision_prompt += "Ensure that the generated Python code includes accurate styling, layout, and positioning while eliminating any unnecessary metamodel-related definitions.\n"
    styling_revision_prompt += "Once you've made the necessary revisions, please provide the updated Python code with the improved layout and styling.\n"



    messages = [
        {
            "role": "system",
            "content": "You are an expert web developer specializing in HTML, CSS, and Python GUI "
            "frameworks. Improve the given Python code to ensure that it includes all the styling and layout aspects."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": styling_revision_prompt
                },
                {
                    "type": "text",
                    "text": python_code
                },
                {
                    "type": "text",
                    "text": code_file_content  # Pass the code as text
                },
                {
                    "type": "text",
                    "text": styling_file_content  # Pass the styling file as text
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
                        "text": metamodel_text_content
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
        timeout=120
        )

    response_json = response.json()


    try:
        improved_code = response_json['choices'][0]['message']['content']
        return improved_code
    except KeyError:
        return None


# Function to enhance the Python GUI code with styling features (layout, color, size)
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_styling_prompting(api_key, code_file_path, base64_metamodel,
                           metamodel_text_path, python_code, styling_file_path):

    # Encode the image
    base64_metamodel = encode_image(metamodel_image_path)

    # Read input files
    code_file_content = read_file_contents(code_file_path)
    metamodel_text_content = read_file_contents(metamodel_text_path) if metamodel_text_path else ""
    styling_file_content = read_file_contents(styling_file_path)

    # Define request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the system instruction
    system_instruction = (
        "You are an expert web developer specializing in HTML, CSS, and Python GUI frameworks. "
        "You will receive Python GUI code that adheres to a specific GUI metamodel, along with the corresponding HTML source code and styling file.. "
        "Your task is to complete this Python code by incorporating layout, size, color, and positioning details. "
        "Ensure that the generated structure adheres to the provided styling file and maintains consistency with the GUI metamodel."
    )

    # Define user instructions
    user_prompt = (
        "### Task: Enhance Python GUI Code with Layout and Styling Features\n"
        "Your task is to refine the given Python GUI code by incorporating essential layout and styling attributes, "
        "Ensuring compliance with the GUI metamodel, including styling and layout metaclasses and their relationships.\n\n"
        "**The provided code follows the GUI metamodel but lacks key attributes such as size, color, positioning, and overall layout.**\n"
        "Modify the Python code to integrate these aspects based on the given HTML source code and styling CSS file.\n\n"
        "**Key Considerations:**\n"
        "- **Layout & Positioning:** Ensure proper alignment, spacing, and responsiveness.\n"
        "- **Size:** Adjust width, height, and font sizes appropriately.\n"
        "- **Color:** Apply the color scheme.\n"
        "- **Consistency:** Maintain alignment with the GUI metamodel structure.\n\n"
        "### Output Requirement:\n"
        "Return only the updated Python code without explanations or additional commentary."
    )


    # Construct the request messages
    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": "### The Python GUI Code:\n" + python_code},
                {"type": "text", "text": "### Corresponding HTML source code:\n" + code_file_content},
                {"type": "text", "text": "### CSS/Styling code:\n" + styling_file_content},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_metamodel}", "detail": "high"}}
            ]
        }
    ]

    # Add metamodel description if available
    if metamodel_text_content:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "### GUI Metamodel Explanation:\n" + metamodel_text_content}
                ]
            }
        )

    # Define the API request payload
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    # Send the request to OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
        )

    # Parse the response
    response_json = response.json()

    # Extract the improved Python code
    try:
        improved_code = response_json['choices'][0]['message']['content']
        return improved_code
    except KeyError:
        print("Error: Unable to retrieve improved Python code from the response.")
        return None


# Function to facilitate self-improvement of a Python code representing GUI elements using an LLM model
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement(api_key, code_file_path, base64_metamodel,
                          metamodel_text_path, python_code,
                          structural_model_path):

    # Encode the image
    base64_metamodel = encode_image(metamodel_image_path)

    # Read file
    code_file_content = read_file_contents(code_file_path)
    metamodel_text_content = read_file_contents(metamodel_text_path)
    structural_model_content = read_file_contents(structural_model_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    ## revision_prompt
    revision_prompt = ""
    revision_prompt += "As an expert Web GUI developer, I need your assistance in improving a Python code that represents the GUI elements of an HTML source code file.\n"
    revision_prompt += "The provided Python code follows the GUI metamodel but may contain missing, incorrect, or misaligned elements related to layout or styling when compared to the original source code file.\n"
    revision_prompt += "I have provided you with the GUI metamodel image file and a description file to assist you in the revision process.\n"
    revision_prompt += "The Python code you receive as input is related to one of the pages of an application designed for performing database operations (CRUD) on a structure of entities.\n"
    revision_prompt += "This structure is defined in the structural model by Python code, representing the entities, their attributes, and the relationships between them.\n"
    revision_prompt += "Please note that in this page, all attributes or a subset of the attributes of the desired class may be considered. You can refer to the list of attributes of the class according to the structural model and incorporate these attributes in the generation of Python code.\n\n"
    revision_prompt += "Your task is to carefully compare the **GUI elements, along with their styling and layout, in the HTML source file** to those in the Python code, and make any necessary revisions.\n"
    revision_prompt += "Please remove any parts of the code that are related to the metaclasses definition of metamodel.\n"
    revision_prompt += "Also, please specify the 'is_main_page=True' attribute for main page of the application.\n"
    revision_prompt += "Once you have revised the Python code, please respond with the updated version.\n"
    revision_prompt += "In the end of the python code that you generated, please mention which class this page is related to in the structural model. Additionally, provide a list of attributes of this class based on the structural model."

    messages = [
        {
            "role": "system",
            "content": "You are a developer. Improve the given Python code to ensure that it includes "
            "all the GUI elements from the HTML source code file with their styling and layout."
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
                    "type": "text",
                    "text": code_file_content  # Pass the code as text
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
                        "text": metamodel_text_content
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
                        "text": structural_model_content
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
        timeout=120
        )

    response_json = response.json()


    try:
        improved_code = response_json['choices'][0]['message']['content']
        return improved_code
    except KeyError:
        return None


# Function to call LLM with the direct prompt method
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4o_call(api_key, code_file_path, base64_metamodel, direct_prompt,
               first_example_source_code_path, second_example_source_code_path,
               first_example_gui_code_path, second_example_gui_code_path,
               metamodel_text_path, structural_model_path):


    # Encode the image
    base64_metamodel = encode_image(metamodel_image_path)

    # Read files
    code_file_content = read_file_contents(code_file_path)
    metamodel_text_content = read_file_contents(metamodel_text_path)

    # Examples
    first_example_source_code_content = read_file_contents(first_example_source_code_path)
    second_example_source_code_content = read_file_contents(second_example_source_code_path)
    first_example_gui_code_content = read_file_contents(first_example_gui_code_path)
    second_example_gui_code_content = read_file_contents(second_example_gui_code_path)


    # Structural model
    structural_model_content = read_file_contents(structural_model_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a developer. Generate Python code that represents the GUI elements "
            "in the provided HTML source code file while conforming to the GUI metamodel."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": direct_prompt
                },
                {
                    "type": "text",
                    "text": code_file_content  # Pass the code as text
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

    if first_example_source_code_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the first example of a code file and the expected Python output:"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the code at this file:\n{first_example_source_code_content}"
                    },
                    {
                        "type": "text",
                        "text": first_example_gui_code_content
                    },
                ]
            }
        )
    if second_example_source_code_path:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the second example of a code file and the expected Python output:"
                    },
                    {
                        "type": "text",
                        "text": f"You can view the code at this file:\n{second_example_source_code_content}"
                    },
                    {
                        "type": "text",
                        "text": second_example_gui_code_content
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
                        "text": metamodel_text_content
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
                        "text": structural_model_content
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
        timeout=120
        )

    response_json = response.json()


    try:
        python_code = response_json['choices'][0]['message']['content']
        return python_code
    except KeyError:
        return None


# Function to send a direct prompt to LLM for generating Python code
def direct_prompting(api_key, metamodel_image_path, code_file_path, first_example_source_code_path,
                     second_example_source_code_path, first_example_gui_code_path, second_example_gui_code_path,
                     metamodel_text_path, structural_model_path):


    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# HTML Source code to Python UI Code Generation\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate Python code that represents the GUI elements of a given HTML source code file and conforms to the GUI metamodel.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "The HTML source code file model you receive as input is related to one of the pages of a web application designed for performing database operations (CRUD) on a structure of entities.\n"
    direct_prompt += "This structure is defined in the structural model by Python code, representing the entities, their attributes, and the relationships between them.\n"
    direct_prompt += "This HTML source code file is related to a page of the app that corresponds to one of the classes in the structural model.\n"
    direct_prompt += "Please note that in this page, all attributes or a subset of the attributes of the desired class may be considered. You can refer to the list of attributes of the class according to the structural model and incorporate these attributes in the generation of Python code.\n\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. I will provide you with example HTML source code files and their expected Python output.\n"
    direct_prompt += "2. I will also provide you with an HTML source code file as input.\n"
    direct_prompt += "3. I will also provide you with an image of the GUI metamodel and a description file for it.\n"
    direct_prompt += "4. Your goal is to generate Python code that accurately represents all GUI elements in the HTML source code file.\n"
    direct_prompt += "5. Please provide the Python code representing the GUI elements only, without any introductory or formatting lines.\n"
    direct_prompt += "6. Pay attention to layout and styling features like size, position, and color of all the elements, as well as the overall layout.\n"
    direct_prompt += "7. It is crucial: the structure of the Python code you generate should be similar to the code samples provided as examples, while the content should be derived from the HTML source code file you receive.\n"
    direct_prompt += "8. The generated code should adhere to the GUI metamodel.\n"
    direct_prompt += "9. Incorporate The provided examples and ensure the generated code aligns with the user's expectations.\n"
    direct_prompt += "10. Once you're ready, I will present you with an HTML source code file, and you need to generate Python code for its GUI elements.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Respond with the Python code that represents the GUI elements of the provided source code file.\n"

    # Encode the image
    base64_metamodel = encode_image(metamodel_image_path)


    # Call gpt4o_call to generate the Python code using the direct prompt method
    python_code = gpt4o_call(api_key, code_file_path, base64_metamodel, direct_prompt,
                             first_example_source_code_path, second_example_source_code_path,
                             first_example_gui_code_path, second_example_gui_code_path,
                             metamodel_text_path, structural_model_path)


    return python_code


def run_pipeline_gui_generation(api_key, code_file_path, output_folder: str, styling_file_path:str=None):

    structural_dir = os.path.join(output_folder, "buml")
    structural_model_path = os.path.join(structural_dir, "model.py")

    # Generate the Python code using the direct prompt method
    python_code = direct_prompting(
        api_key,
        metamodel_image_path,
        code_file_path,
        first_example_source_code_path,
        second_example_source_code_path,
        first_example_gui_code_path,
        second_example_gui_code_path,
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
            gui_output_dir, f"{get_file_name(code_file_path)}.py"
        )
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(python_code)
        #print(f"Generated Python code saved to {output_file_name}")
    else:
        print("Failed to generate Python code.")

    # Generate the revised Python code using the self-improvement method
    improved_code = gpt4_self_improvement(api_key, code_file_path, metamodel_image_path,
                                          metamodel_text_path, python_code, structural_model_path)

    # Save the generated revised code to a file
    if improved_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py")
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(improved_code)
        #print(f"Generated revised code saved to {output_file_name}")
    else:
        print("Failed to generate revised code.")

    final_code = gpt4_fix_string_properties (api_key, improved_code, structural_model_path)

    # Save the generated code to a file
    if final_code:
        output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py") # Specify the desired output file name
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(final_code)
        print(f"Generated complete code saved to {output_file_name}")
    else:
        print("Failed to generate revise code.")


    if styling_file_path:
        styling_code = gpt4_styling_prompting(api_key, code_file_path, metamodel_image_path,
                                              metamodel_text_path, final_code, styling_file_path)

        if styling_code:
            output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py")
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(styling_code)
            #print(f"Generated revised code saved to {output_file_name}")
        else:
            print("Failed to generate code.")


    if styling_file_path:
        # Generate the revised Python code using the self-improvement method
        improved_styling_code = gpt4_self_improvement_styling_part(api_key, code_file_path, metamodel_image_path,
                                                                   metamodel_text_path, styling_code, styling_file_path)

        # Save the generated revised code to a file
        if improved_styling_code:
            output_file_name = os.path.join(gui_output_dir, "generated_gui_model.py")
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(improved_styling_code)
            #print(f"Generated revised code saved to {output_file_name}")
        else:
            print("Failed to generate revised code.")

