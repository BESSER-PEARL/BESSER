import os
import retrying
import requests
from besser.BUML.notations.sourceCode_to_structural.utilities.file_utils import read_file_contents
from besser.BUML.notations.sourceCode_to_structural.config import first_code_file_example_path, plantuml_code_example_path


@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def gpt4_self_improvement_plantuml_code(api_key, plantuml_code_content):


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    revision_prompt = (
    "As an expert in modeling with PlantUML, I need your assistance in improving a PlantUML code "
    "that represents the structure of a single web page.\n\n"
    "⚠️ VERY IMPORTANT:\n"
    "1. Ensure that the PlantUML code does NOT include any associations with duplicate names. "
    "Each association must have a unique identifier.\n"
    "2. Each web page should correspond to ONE class in PlantUML. Do not create multiple classes for one page.\n"
    "3. Remove all method definitions from the classes. Only keep attributes.\n"
    "4. VERY IMPORTANT: Consider ONLY the following attribute types: int, float, str, bool, time, date, datetime, timedelta. "
    "Do NOT use any other types under any circumstances.\n"
    "5. Keep the PlantUML code syntactically correct, starting with '@startuml' and ending with '@enduml'.\n"
    "6. Do not create duplicate attribute names inside the class.\n\n"
    "Please revise the provided PlantUML code according to these rules and respond with "
    "the updated PlantUML code ONLY, without explanations."
)


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
def gpt4o_call_plantuml(api_key, prompt, source_code_content, first_code_file_example_path, plantuml_code_example_path):


    # Read the content of example
    first_code_file_example_path = read_file_contents(first_code_file_example_path)
    plantuml_code_example = read_file_contents(plantuml_code_example_path)


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
       {
           "role": "system",
           "content": "You are a PlantUML developer. Generate PlantUML code (PlantUML class diagram) from the provided code file that reflects the complete UI design of the web page."
       },
       {
           "role": "user",
           "content": [
               {
                   "type": "text",
                   "text": prompt
               },
                {
                        "type": "text",
                        "text": source_code_content
                },
           ]
       }
   ]

    if first_code_file_example_path:
        messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Here are the example for a code file of one web page:"
                },
                {
                    "type": "text",
                    "text": f"You can view the UI code for the web page example: \n{first_code_file_example_path}"
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


    try:
        plantuml_code = response_json['choices'][0]['message']['content']
        return plantuml_code
    except KeyError:
        return None


def direct_prompting_source_code_to_plantuml(api_key, source_code_content, first_code_file_example_path, plantuml_code_example_path):


    # Direct Prompt
    direct_prompt = ""
    direct_prompt += "# Prompt:\n"
    direct_prompt += "# Generate PlantUML Code for Application UI Code\n\n"

    direct_prompt += "## Task:\n"
    direct_prompt += "Generate a PlantUML diagram representing the structural model of a web application's UI, based on a provided source code file for a single page.\n\n"

    direct_prompt += "## Description:\n"
    direct_prompt += "You will receive a source code file representing a web application page. Your task is to analyze the code and generate a corresponding PlantUML model that accurately captures its structure.\n"
    direct_prompt += "This code file corresponds to a single class in the PlantUML diagram, representing one page of the web app.\n\n"

    direct_prompt += "## Instructions:\n"
    direct_prompt += "1. You will be given example code file and its corresponding PlantUML code, demonstrating how each UI component maps to the application's structural model.\n"
    direct_prompt += "2. Using the provided code file (and example for guidance), generate a PlantUML diagram that represents the page's structure.\n"
    direct_prompt += "3. Ensure the generated PlantUML dose not have any syntax error according to PlantUML code.\n"
    direct_prompt += "4. **Strictly avoid assumptions or additional functionality** beyond what is present in the source code file.\n"
    direct_prompt += "5. Make sure to consider the following attribute types: int, float, str, bool, time, date, datetime, and timedelta in resulting PlantUML code.\n"
    direct_prompt += "6. Avoid duplicate association names in the PlantUML diagram.\n"
    direct_prompt += "7. Do not include methods within class definitions in the PlantUML diagram.\n"
    direct_prompt += "8. Buttons should not be treated as attributes within any class.\n"
    direct_prompt += "9. Ensure the generated PlantUML includes all relevant classes, attributes, and relationships between components, accurately reflecting the entire application's design as seen in the code file.\n\n"

    direct_prompt += "## Expected Output:\n"
    direct_prompt += "Using the provided code file and example PlantUML code, generate a single PlantUML file that encapsulates the entire application model, including classes, attributes, and relationships.\n"


    plantuml_code = gpt4o_call_plantuml(api_key, direct_prompt, source_code_content, first_code_file_example_path, plantuml_code_example_path)


    return plantuml_code


def run_pipeline_plantuml_generation(api_key: str, source_code_content: str, output_folder: str):


    # Generate the code using the direct prompt method
    plantuml_code = direct_prompting_source_code_to_plantuml(
        api_key, source_code_content, first_code_file_example_path, plantuml_code_example_path
    )

    # Define the plantuml subfolder inside the output folder
    plantuml_folder = os.path.join(output_folder, "plantuml")
    output_file_name = os.path.join(plantuml_folder, "generated_plantuml.puml")

    # Save the generated code to a file
    if plantuml_code:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding='utf-8') as file:
            file.write(plantuml_code)
    else:
        print("Failed to generate PlantUML code.")


    # Generate the revised code using the self-improvement method
    improved_plantuml_code = gpt4_self_improvement_plantuml_code(api_key, plantuml_code)

    # Save the revised code to a file
    if improved_plantuml_code:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding='utf-8') as file:
            file.write(improved_plantuml_code)
        #print(f"Generated PlantUML code saved to {output_file_name}")
    else:
        print("Failed to generate revised code.")

