import os
import requests
import retrying
from besser.BUML.notations.sourceCode_to_buml.utilities import get_file_name


# Function to enhance the generated SVG based on Norman’s Desirability Principle using GPT-4o
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def enhance_svg_desirability(api_key, svg_code):
    """
    Enhances the given SVG code by improving its visual appeal based on Norman’s Desirability principle.
    Uses GPT-4o to refine the design for aesthetics, inspiration, and user emotional impact.

    Args:
        api_key (str): OpenAI API key.
        svg_code (str): Original SVG content to be optimized.

    Returns:
        str: Improved SVG code or None if generation fails.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Prompt for improving Desirability
    desirability_prompt = (
        "You are a UI optimization assistant. Improve the following SVG code by enhancing its Desirability "
        "based on Norman’s UX principles. Make the UI visually appealing and inspiring.\n\n"
        "Apply the following:\n"
        "1. Use elegant and harmonious color schemes.\n"
        "2. Improve typography (font size, weight, style).\n"
        "3. Add visual polish such as subtle shadows, rounded corners, or gradients.\n"
        "4. Ensure balanced spacing and layout aesthetics.\n"
        "⚠️ IMPORTANT:\n"
        "- Preserve ALL existing `id` attributes exactly as they are in the input SVG.\n"
        "- If an element already has an `id`, keep it unchanged.\n"
        "- Do NOT rename, remove, or generate new IDs.\n"
        "- If you add new elements that require IDs, prefix them with 'gen_' to distinguish them.\n"
        "\n"
        "Return ONLY the enhanced SVG code, no explanations."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a UI design assistant. Improve the visual appeal of SVG code for desirability."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": desirability_prompt},
                {"type": "text", "text": svg_code}
            ]
        }
    ]

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
            )

        response_json = response.json()
        return response_json.get('choices', [{}])[0].get('message', {}).get('content', None)

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None


# Function to enhance the generated SVG based on Norman’s Findability Principle using GPT-4o
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def enhance_svg_findability(api_key, svg_code):
    """
    Enhances the given SVG code by improving findability, ensuring that users can quickly locate essential elements.
    Uses GPT-4o to refine UI elements based on Norman’s Findability principle.

    Args:
        api_key (str): OpenAI API key.
        svg_code (str): SVG content to be optimized.

    Returns:
        str: Improved SVG code or None if generation fails.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Structured prompt for findability improvement
    findability_prompt = (
        "You are a UI optimization assistant. Your task is to analyze the provided SVG file "
        "and improve its findability by making key UI elements easy to locate and access."
        "Apply the following principles:\n"
        "1. Improve visual hierarchy: Use size, contrast, and alignment to highlight key actions.\n"
        "2. Enhance navigation: Ensure clear and distinct navigation bars, menus, and buttons.\n"
        "3. Organize content logically: Group related elements together for easier discovery.\n"
        "4. Use recognizable icons and labels: Help users quickly identify important sections.\n"
        "5. Maintain consistent element placement: Place buttons, menus, and forms in expected positions.\n"
        "6. Ensure sufficient whitespace: Prevent clutter and improve scanability.\n"
        "⚠️ IMPORTANT:\n"
        "- Preserve ALL existing `id` attributes exactly as they are in the input SVG.\n"
        "- If an element already has an `id`, keep it unchanged.\n"
        "- Do NOT rename, remove, or generate new IDs.\n"
        "- If you add new elements that require IDs, prefix them with 'gen_' to distinguish them.\n"
        "\n"
        "Return only the improved SVG code without any explanations."
    )

    # Construct the API request messages
    messages = [
        {
            "role": "system",
            "content": "You are a UI optimization assistant. Improve the given "
            "SVG code to enhance findability and navigation clarity."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": findability_prompt},
                {"type": "text", "text": svg_code}
            ]
        }
    ]

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
            )

        response_json = response.json()

        return response_json.get('choices', [{}])[0].get('message', {}).get('content', None)

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None


# Function to enhance the generrated SVG based on Norman’s usability Principle using GPT-4o
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def enhance_svg_usability(api_key, svg_code):
    """
    Enhances the given SVG code by improving its usability, ensuring simplicity, intuitive navigation, and efficiency.
    Uses GPT-4o to refine UI elements based on Norman’s Usability principle.

    Args:
        api_key (str): OpenAI API key.
        svg_code (str): SVG content to be optimized.

    Returns:
        str: Improved SVG code or None if generation fails.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Structured prompt for usability improvement
    usability_prompt = (
        "You are a UI optimization assistant. Your task is to analyze the provided SVG file "
        "and improve its usability, ensuring that it is simple and easy to use."
        "Apply the following principles:\n"
        "1. Simplify UI interactions: Ensure intuitive buttons, forms, and labels.\n"
        "2. Improve readability and accessibility: Use legible font sizes and proper contrast.\n"
        "3. Enhance navigation clarity: Organize elements logically for smooth navigation.\n"
        "4. Reduce cognitive load: Minimize user effort by streamlining key actions.\n"
        "5. Provide immediate feedback and guidance: Implement hover states and clear messages.\n"
        "6. Ensure consistency: Use uniform colors, fonts, and layouts.\n"
        "⚠️ IMPORTANT:\n"
        "- Preserve ALL existing `id` attributes exactly as they are in the input SVG.\n"
        "- If an element already has an `id`, keep it unchanged.\n"
        "- Do NOT rename, remove, or generate new IDs.\n"
        "- If you add new elements that require IDs, prefix them with 'gen_' to distinguish them.\n"
        "\n"
        "Return only the improved SVG code without any explanations."
    )


    # Construct the API request messages
    messages = [
        {
            "role": "system",
            "content": "You are a UI optimization assistant. Improve the "
            "given SVG code to enhance usability and user experience."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": usability_prompt},
                {"type": "text", "text": svg_code}
            ]
        }
    ]


    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
            )

        response_json = response.json()

        return response_json.get('choices', [{}])[0].get('message', {}).get('content', None)

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None



# Function to enhance SVG usefulness based on Norman’s Usefulness Principle using GPT-4o
@retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
def enhance_svg_usefulness(api_key, svg_code):
    """
    Enhances the given SVG code by improving its usefulness, ensuring clarity, relevance, and task efficiency.
    Uses GPT-4o to refine UI elements based on Norman’s Usefulness principle.

    Args:
        api_key (str): OpenAI API key.
        svg_code (str): SVG content to be optimized.

    Returns:
        str: Improved SVG code or None if generation fails.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Structured prompt for usefulness improvement
    usefulness_prompt = (
        "You are a UI optimization assistant. Your task is to analyze the given SVG file "
        "and improve its usefulness by ensuring clarity, relevance, and task efficiency. "
        "Apply the following principles:\n"
        "1. Remove unnecessary UI elements that do not contribute to the user’s primary goals.\n"
        "2. Ensure key actions and navigation elements are easily identifiable and accessible.\n"
        "3. Improve the organization of elements to guide users naturally toward important tasks.\n"
        "4. Maintain a consistent color scheme, font style, and spacing to reinforce clarity.\n"
        "5. Use grid-based layouts or proper alignment to structure content logically and intuitively.\n"
        "6. Optimize spacing between elements to prevent visual clutter while maintaining readability.\n"
        "⚠️ IMPORTANT:\n"
        "- Preserve ALL existing `id` attributes exactly as they are in the input SVG.\n"
        "- If an element already has an `id`, keep it unchanged.\n"
        "- Do NOT rename, remove, or generate new IDs.\n"
        "- If you add new elements that require IDs, prefix them with 'gen_' to distinguish them.\n"
        "\n"
        "Return only the improved SVG code without additional explanations."
    )

    # Construct the API request messages
    messages = [
        {
            "role": "system",
            "content": "You are a UI optimization assistant. Improve the "
            "given SVG code to enhance the usefulness and user experience."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": usefulness_prompt},
                {"type": "text", "text": svg_code}
            ]
        }
    ]

    # Prepare the API request payload
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
            )

        response_json = response.json()

        # Extract and return improved SVG code
        return response_json['choices'][0]['message']['content']

    except KeyError:
        print("Error: Failed to retrieve improved SVG content from response.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None


def run_pipeline_svg_enhancement(api_key: str, output_folder: str):

    output_file_name = None

    svg_output_dir=os.path.join(output_folder, "svg")
    svg_hci_enhance_output_dir=os.path.join(output_folder, "hci_enhanced")

    # Ensure the HCI enhanced output directory exists
    os.makedirs(svg_hci_enhance_output_dir, exist_ok=True)

    # List all files in the folder
    svg_code_file_paths = [
        os.path.join(svg_output_dir, f)
        for f in os.listdir(svg_output_dir)
        if os.path.isfile(os.path.join(svg_output_dir, f))
    ]

    for index, svg_code_file_path in enumerate(svg_code_file_paths, start=1):

        # 🔹 Read the actual SVG content
        with open(svg_code_file_path, "r", encoding="utf-8") as file:
            svg_content = file.read()

        # Enhance for Usefulness
        improved_svg_code_usefulness = enhance_svg_usefulness(api_key, svg_content)

        # Enhance for Usability
        improved_svg_code_usability = enhance_svg_usability(api_key, improved_svg_code_usefulness)

        # Enhance for Findability
        improved_svg_code_findability = enhance_svg_findability(api_key, improved_svg_code_usability)

        # Enhance for Desirability
        improved_svg_code_desirability = enhance_svg_desirability(api_key, improved_svg_code_findability)

        if improved_svg_code_desirability:
            output_file_name = os.path.join(
                svg_hci_enhance_output_dir,
                f"{get_file_name(svg_code_file_path)}.svg"
            )

            # 🔹 Save the FINAL enhanced code (not just usefulness)
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(improved_svg_code_desirability)

            print(f"HCI-enhanced code saved to {output_file_name}")
        else:
            print("❌ Failed to enhance SVG code for HCI.")
            return None

    return output_file_name




