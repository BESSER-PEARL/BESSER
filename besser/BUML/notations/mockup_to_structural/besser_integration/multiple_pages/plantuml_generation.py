import retrying
import requests
import os
from besser.BUML.notations.mockup_to_structural.utilities import encode_image
from besser.BUML.notations.mockup_to_structural.utilities import read_file_contents
from besser.BUML.notations.mockup_to_structural.config import *


# Function to facilitate self-improvement of a PlantUML code representing GUI elements using the GPT-4o model
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement_plantUml(api_key, plantuml_code_content):


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    ## revision_prompt
    revision_prompt = ""
    revision_prompt += "As an expert in modeling with PlantUML, I need your assistance in improving a PlantUML code that represents the structure of a web page.\n"
    revision_prompt += "Improve the name of assocation in the provided PlantUML code to ensure that the PlantUML code does not contain associations with the same names.\n"
    revision_prompt += "Please note that the provided PlantUML code is related to the pages of the web app and each page corresponds to one class in PlantUML. So, in PlantUML code, the structure of UI page must be represented by one class.\n"
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

    # Print the response JSON
    #print("Response JSON:")
    #print(json.dumps(response_json, indent=4))  # Print with indentation for better readability

    try:
        improved_plantuml_code = response_json['choices'][0]['message']['content']
        return improved_plantuml_code
    except KeyError:
        return None



# Function to call GPT-4o with the direct prompt method
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4o_call_palntUML(api_key, prompt, mockup_images_path, additional_text_file_path, plantuml_code_example_path):


    # the code to handle a folder path containing multiple images
    base64_mockup_images = []
    for root,dirs,files in os.walk(mockup_images_path):
        for file in files:
            image_path = os.path.join(root, file)
            base64_image = encode_image(image_path)
            base64_mockup_images.append(base64_image)


    plantuml_code_example = read_file_contents(plantuml_code_example_path)

    additional_text_file_path = read_file_contents(additional_text_file_path)



    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
       {
           "role": "system",
           "content": "You are a PlantUML developer. Generate PlantUML code (PlantUML class diagram) from the provided UI images that reflects the complete UI design of the web page."
       },
       {
           "role": "user",
           "content": [
               {
                   "type": "text",
                   "text": prompt
               }
           ]
       }
   ]
    # Add image URLs for each mockup image
    for base64_mockup_image in base64_mockup_images:
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_mockup_image}",
                "detail": "high"
            }
        }
        messages[1]["content"].append(image_data)


        messages[1]["content"].append(additional_text_file_path)

    if plantuml_code_example_path:
        messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Here are the images of pages of the web app:"
                },
                {
                    "type": "text",
                    "text": "You can view the UI image for the main page of the app: [manin page Image](https://imgurl.ir/uploads/p876089_main_page.png)"
                },
                {
                    "type": "text",
                    "text": "You can view the UI image for the second page of the app: [Second page Image](https://imgurl.ir/uploads/f832119_library_directory.png)"
                },
                {
                    "type": "text",
                    "text": "You can view the UI image for the Third page of the app: [Third page Image](https://imgurl.ir/uploads/x12382_author_list.png)"
                },
                {
                    "type": "text",
                    "text": "You can view the UI image for the fourth page of the app: [Fourth Page](https://imgurl.ir/uploads/c588836_book_list.png)"
                },
                {
                    "type": "text",
                    "text": f"You can view the PlantUML code for all pages of the app: [expected output]\n{plantuml_code_example}"
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
        plantuml_code = response_json['choices'][0]['message']['content']
        return plantuml_code
    except KeyError:
        return None


# Function to send a direct prompt to GPT-4o for generating PlantUML code
def direct_prompting_image_to_plantuml(api_key, mockup_images_path, additional_text_file_path, plantuml_code_example_path):


    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# PlantUML Code Generation for Application Model\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate a PlantUML code that represents the structural model of an application based on provided UI images from several pages of the app.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "You will receive UI images that represent pages of a web application.\n"
    direct_prompt += "Your goal is to analyze the provided UI images and use them to generate a PlantUML model that accurately represents the application's structure in a PlantUML file.\n"
    direct_prompt += "To better understand the associations between various classes within the application structure, additional information as a text file will be provided.\n"
    direct_prompt += "Please note that each UI image corresponds to one class in PlantUML, representing a single page of the web app.\n\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. You will be given example UI images and their corresponding PlantUML code, demonstrating how each UI component maps to the application's structural model.\n"
    direct_prompt += "2. Based on the provided UI images (and the example as guidance), generate a PlantUML file that represents the application's overall structure.\n"
    direct_prompt += "3. Ensure the generated PlantUML dose not have any syntax error according to PlantUML code.\n"
    direct_prompt += "4. Make sure to consider the following attribute types: int, float, str, bool, time, date, datetime, and timedelta.\n"
    direct_prompt += "5. Ensure that the PlantUML code does not contain associations with the same names.\n"
    direct_prompt += "6. Ensure that the PlantUML code does not contain methods within the class definition.\n"
    direct_prompt += "7. Please ensure that buttons are not treated as attributes for any classes in the PlantUML code."
    direct_prompt += "8. Ensure the generated PlantUML includes all relevant classes, attributes, and relationships between components, accurately reflecting the entire application's design as seen in the UI images.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Using the provided UI images and example PlantUML code, generate a single PlantUML file that encapsulates the entire application model, including classes, attributes, and relationships.\n"



    # Call gpt4o_call to generate the PlantUML code using the direct prompt method
    PlantUML_code = gpt4o_call_palntUML(api_key, direct_prompt, mockup_images_path, additional_text_file_path, plantuml_code_example_path)


    return PlantUML_code


def run_pipeline_plantuml_generation(api_key: str, mockup_images_path: str, output_folder: str,
                                     additional_text_file_path: str):

    # Generate the PlantUML code using the direct prompt method
    plantuml_code = direct_prompting_image_to_plantuml(api_key, mockup_images_path, additional_text_file_path,
                                                       plantuml_code_example_path)

    # Define the plantuml subfolder inside the output folder
    plantuml_folder = os.path.join(output_folder, "plantuml")
    # Ensure the plantuml folder exists
    os.makedirs(plantuml_folder, exist_ok=True)

    output_file_name = os.path.join(plantuml_folder, "generated_plantuml.puml")

    # Save the generated code to a file
    if plantuml_code:
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(plantuml_code)
        print(f"Generated PlantUML code saved to {output_file_name}")
    else:
        print("Failed to generate PlantUML code.")


    # Generate the revised PlantUML code using the self-improvement method"
    improved_plantuml_code = gpt4_self_improvement_plantUml(api_key, plantuml_code)


    # Save the generated code to a file
    if improved_plantuml_code:
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(improved_plantuml_code)
        print(f"Generated revise code saved to {output_file_name}")
    else:
        print("Failed to generate revise code.")