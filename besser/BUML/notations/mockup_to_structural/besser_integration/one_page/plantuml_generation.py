import retrying
import requests
from besser.BUML.notations.mockup_to_structural.utilities import encode_image
from besser.BUML.notations.mockup_to_structural.utilities import read_file_contents
from besser.BUML.notations.mockup_to_structural.config import *


@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement_plantuml_code(api_key, plantuml_code_content):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    ## revision_prompt
    revision_prompt = ""
    revision_prompt += "As an expert in modeling with PlantUML, I need your assistance in improving a PlantUML code that represents the structure of a web page.\n"
    revision_prompt += "Improve the name of assocation in the provided PlantUML code to ensure that the PlantUML code does not contain associations with the same names.\n"
    revision_prompt += "Please note that the provided PlantUML code is related to one page of the web app and each page corresponds to one class in PlantUML. So, in PlantUML code, the structure of UI page must be represented by one class.\n"
    revision_prompt += "Please remove any method definition in classes in Provided PlantUML code.\n"
    revision_prompt += "Make sure to consider the following attribute types: int, float, str, bool, time, date, datetime, and timedelta in resulting PlantUML code.\n"
    revision_prompt += "Once you have revised the PlantUML code, please respond with the updated version.\n"

    messages = [
        {
            "role": "system",
            "content": "You are a developer. Improve the given PlantUML code to ensure that it includes the structure of the web app."
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
                    "text": plantuml_code_content
                }
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
        improved_plantuml_code = response_json['choices'][0]['message']['content']
        return improved_plantuml_code
    except KeyError:
        return None



# Function to call GPT-4o with the direct prompt method
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4o_call_plantuml(api_key, prompt, mockup_image_path, first_gui_model_path, plantuml_code_example_path):


    base64_mockup_image = encode_image(mockup_image_path)
    plantuml_code_example = read_file_contents(plantuml_code_example_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
       {
           "role": "system",
           "content": "You are a PlantUML developer. Generate PlantUML code (PlantUML class diagram) from the provided UI image that reflects the complete UI design of the web page."
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
                       "url": f"data:image/jpeg;base64,{base64_mockup_image}",
                       "detail": "high"
                   }
               }
           ]
       }
   ]

    if first_gui_model_path:
        messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Here are the image of one web page:"
                },
                {
                    "type": "text",
                    "text": "You can view the UI image for the web page: [page Image](https://imgurl.ir/uploads/f832119_library_directory.png)"
                },
                {
                    "type": "text",
                    "text": f"You can view the PlantUML code for the web page: [expected output]\n{plantuml_code_example}"
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


    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

    try:
        plantuml_code = response_json['choices'][0]['message']['content']
        return plantuml_code
    except KeyError:
        return None


def direct_prompting_image_to_plantuml(api_key, mockup_image_path, first_gui_model_path, plantuml_code_example_path):

    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# PlantUML Code Generation for Application Model\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate a PlantUML code that represents the structural model of an application based on provided UI image from one page of the app.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "You will receive UI image that represent one page of a web application.\n"
    direct_prompt += "Your goal is to analyze the provided UI image and use it to generate a PlantUML model that accurately represents the application's structure in a PlantUML file.\n"
    direct_prompt += "Please note that each UI image corresponds to one class in PlantUML, representing a single page of the web app.\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. You will be given example UI image and its corresponding PlantUML code, demonstrating how each UI component maps to the application's structural model.\n"
    direct_prompt += "2. Based on the provided UI image (and the example as guidance), generate a PlantUML file that represents the web page's structure.\n"
    direct_prompt += "3. Ensure the generated PlantUML dose not have any syntax error according to PlantUML code.\n"
    direct_prompt += "4. Make sure to consider the following attribute types: int, float, str, bool, time, date, datetime, and timedelta in resulting PlantUML code.\n"
    direct_prompt += "5. Ensure that the PlantUML code does not contain associations with the same names.\n"
    direct_prompt += "6. Ensure that the PlantUML code does not contain methods within the class definition.\n"
    direct_prompt += "7. Please ensure that buttons are not treated as attributes for any classes in the PlantUML code."
    direct_prompt += "8. Ensure the generated PlantUML includes all relevant classes, attributes, and relationships between components, accurately reflecting the entire application's design as seen in the UI image.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Using the provided UI images and example PlantUML code, generate a single PlantUML file that encapsulates the entire application model, including classes, attributes, and relationships.\n"



    plantuml_code = gpt4o_call_plantuml(api_key, direct_prompt, mockup_image_path,
                                        first_gui_model_path, plantuml_code_example_path)


    return plantuml_code


def run_pipeline_plantuml_generation(api_key: str, mockup_image_path: str, output_folder: str):

    # Generate the code using the direct prompt method
    plantuml_code = direct_prompting_image_to_plantuml(
        api_key, mockup_image_path, first_gui_model_path, plantuml_code_example_path
    )

    # Define the plantuml subfolder inside the output folder
    plantuml_folder = os.path.join(output_folder, "plantuml")
    output_file_name = os.path.join(plantuml_folder, "generated_plantuml.puml")

    # Save the generated code to a file
    if plantuml_code:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(plantuml_code)
    else:
        print("Failed to generate PlantUML code.")

    # Generate the revised code using the self-improvement method
    improved_plantuml_code = gpt4_self_improvement_plantuml_code(api_key, plantuml_code)

    # Save the revised code to a file
    if improved_plantuml_code:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(improved_plantuml_code)
    else:
        print("Failed to generate revised code.")
