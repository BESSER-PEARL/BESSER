import os

#Read the contents of the Python file
def read_file_contents(file_path):
    with open(file_path, "r") as file:
        file_contents = file.read()
    return file_contents

# Function to get the file name without the extension
def get_file_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]
