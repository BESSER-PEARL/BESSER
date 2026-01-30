
import os
import retrying
import requests
from besser.BUML.notations.sourceCode_to_structural.utilities.file_utils import read_file_contents
from besser.BUML.notations.sourceCode_to_structural.config import \
    first_code_file_example_path, second_code_file_example_path, third_code_file_example_path, \
    fourth_code_file_example_path, plantuml_multiple_pages_code_example_path



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

    try:
        improved_plantuml_code = response_json['choices'][0]['message']['content']
        return improved_plantuml_code
    except KeyError:
        return None


# Function to call GPT-4o with the direct prompt method
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4o_call_palntUML(api_key, prompt, source_code_files_path, additional_text_file_path, first_code_file_example_path, second_code_file_example_path, third_code_file_example_path, fourth_code_file_example_path, plantuml_multiple_pages_code_example_path):


    # the code to handle a folder path containing source code files
    source_code_files_contnet = []
    for root, dirs, files in os.walk(source_code_files_path):
        for file in files:
            source_code_path = os.path.join(root, file)
            source_code_file_content = read_file_contents(source_code_path)
            source_code_files_contnet.append(source_code_file_content)

    # Read files
    first_code_file_example_content = read_file_contents(first_code_file_example_path)
    second_code_file_example_content = read_file_contents(second_code_file_example_path)
    third_code_file_example_content = read_file_contents(third_code_file_example_path)
    fourth_code_file_example_content = read_file_contents(fourth_code_file_example_path)
    plantuml_multiple_pages_code_example_content = read_file_contents(plantuml_multiple_pages_code_example_path)

    additional_text_file_content = read_file_contents(additional_text_file_path)



    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
       {
           "role": "system",
           "content": "You are a PlantUML developer. Generate PlantUML code (PlantUML class diagram) from the provided source code files that reflects the complete UI design of the web page."
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
    # Add code content for each source code file
    for source_code_file_contnet in source_code_files_contnet:
        code_content = {
            "type": "text",
            "text": source_code_file_contnet

        }
        messages[1]["content"].append(code_content)


    if plantuml_multiple_pages_code_example_path:
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
                    "text": f"You can view the source code file for the main page of the app at this file: [manin page source code file] \n{fourth_code_file_example_content}"
                },
                {
                    "type": "text",
                    "text": f"You can view the source code file for the second page of the app: [Second page source code file] \n{first_code_file_example_content}"
                },
                {
                    "type": "text",
                    "text": f"You can view the source code file for the third page of the app: [Third page source code file] \n{second_code_file_example_content}"
                },
                {
                    "type": "text",
                    "text": f"You can view the source code file for the fourth page of the app: [Fourth page source code file] \n{third_code_file_example_content}"
                },
                {
                    "type": "text",
                    "text": f"You can view the PlantUML code for all pages of the app: [expected output]\n{plantuml_multiple_pages_code_example_content}"
                }
            ]
        }
    )
    if additional_text_file_path:
        messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Here are the additional file to To understand the associations between various classes within the application structure:"
                },
                {
                    "type": "text",
                    "text": f"You can view the additional file content at this file: \n{additional_text_file_content}"
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

    # Delay to give the model time to think
    #time.sleep(5)  # Adjust the delay duration as needed

    response_json = response.json()


    try:
        plantuml_code = response_json['choices'][0]['message']['content']
        return plantuml_code
    except KeyError:
        return None


# Function to send a direct prompt to GPT-4o for generating PlantUML code
def direct_prompting_image_to_plantuml(api_key, source_code_files_path, additional_text_file_path):


    #direct_prompt:
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# PlantUML Code Generation for Application source code base\n\n"
    direct_prompt += "## Task:\n"
    direct_prompt += "Your task is to generate a PlantUML code that represents the structural model of an application based on provided source code files from several pages of the app.\n\n"
    direct_prompt += "## Description:\n"
    direct_prompt += "You will receive source code files that represent pages of a web application.\n"
    direct_prompt += "Your goal is to analyze the provided source code files and use them to generate a PlantUML model that accurately represents the application's structure in a PlantUML file.\n"
    direct_prompt += "To better understand the associations between various classes within the application structure, additional information as a text file will be provided.\n"
    direct_prompt += "Each source code file should map to a single class in PlantUML, representing one page of the web application. Do not include general page such as the homepage or main page as classes in the PlantUML diagram.\n\n"
    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. You will be given example source code files and their corresponding PlantUML code, demonstrating how each UI component maps to the application's structural model.\n"
    direct_prompt += "2. Based on the provided source code files (and the example as guidance), generate a PlantUML file that represents the application's overall structure.\n"
    direct_prompt += "3. **Strictly avoid assumptions or additional functionality** beyond what is present in the source code files.\n"
    direct_prompt += "4. Ensure the generated PlantUML dose not have any syntax error according to PlantUML code.\n"
    direct_prompt += "5. Make sure to consider the following attribute types: int, float, str, bool, time, date, datetime, and timedelta.\n"
    direct_prompt += "6. Ensure that the PlantUML code does not contain associations with the same names.\n"
    direct_prompt += "7. Ensure that the PlantUML code does not contain methods within the class definition.\n"
    direct_prompt += "8. Please ensure that buttons are not treated as attributes for any classes in the PlantUML code."
    direct_prompt += "9. Ensure the generated PlantUML includes all relevant classes, attributes, and relationships between components, accurately reflecting the entire application's design as seen in the source code files.\n\n"
    direct_prompt += "## Your Task:\n"
    direct_prompt += "Using the provided source code files and example PlantUML code, generate a single PlantUML file that encapsulates the entire application model, including classes, attributes, and relationships.\n"



    # Call gpt4o_call to generate the PlantUML code using the direct prompt method
    PlantUML_code = gpt4o_call_palntUML(api_key, direct_prompt, source_code_files_path, additional_text_file_path, first_code_file_example_path, second_code_file_example_path, third_code_file_example_path, fourth_code_file_example_path, plantuml_multiple_pages_code_example_path)


    return PlantUML_code



def run_pipeline_plantuml_generation(api_key: str, source_code_files_path: str,
                                     output_folder: str, additional_text_file_path: str):

    # Generate the PlantUML code using the direct prompt method
    plantuml_code = direct_prompting_image_to_plantuml(api_key, source_code_files_path, additional_text_file_path)


    # Define the plantuml subfolder inside the output folder
    plantuml_folder = os.path.join(output_folder, "plantuml")
    # Ensure the plantuml folder exists
    os.makedirs(plantuml_folder, exist_ok=True)

    output_file_name = os.path.join(plantuml_folder, "generated_plantuml.puml")

    # Save the generated code to a file
    if plantuml_code:
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)  # Specify the desired output file name
        with open(output_file_name, "w", encoding='utf-8') as file:
            file.write(plantuml_code)
        print(f"Generated PlantUML code saved to {output_file_name}")
    else:
        print("Failed to generate PlantUML code.")


    # Generate the revised PlantUML code using the self-improvement method"
    improved_plantuml_code = gpt4_self_improvement_plantUml(api_key, plantuml_code)


    # Save the generated code to a file
    if improved_plantuml_code:
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding='utf-8') as file:
            file.write(improved_plantuml_code)
        print(f"Generated revise code saved to {output_file_name}")
    else:
        print("Failed to generate revise code.")
