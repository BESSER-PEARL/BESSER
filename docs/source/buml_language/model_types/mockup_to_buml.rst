From UI Mock-ups to B-UML 
===========================


BESSER introduces an innovative low-code approach to automating the extraction of B-UML models, encompassing both structural and GUI representations, directly from 
mockups. Leveraging the power of Large Language Models (LLMs), BESSER transforms visual UI representations into functional software artifacts, 
streamlining the design and development of web applications.


Key Contributions
-----------------
Below are the key functionalities of this component of BESSER:

1. Structural Model Extraction:
    - Automatically derives Structural model from UI mock-up images.
    - :doc:`Learn more about the Structural Model <structural>`

2. GUI Model Extraction:
    - Generates IFML-like GUI model aligned seamlessly with structural models.
    - :doc:`Learn more about the GUI Model <gui>`
    

.. image:: ../..//img/mockup_to_buml.png
  :width: 700
  :alt: From UI mock-ups to B-UML model
  :align: center

For each phase of the process, we have implemented task-specific prompts. You can explore the details for each step by clicking on the relevant links below:


- **Step 1:** Explore the implementation for deriving structural models in our `mockup_to_buml` component. (`GitHub Repository <https://github.com/BESSER-PEARL/BESSER/tree/feature/mockup_to_buml/besser/BUML/notations/mockup_to_buml>`__)
- **Step 2:** Learn about the generation of GUI models, including handling multiple pages, in our `mockup_to_gui` component. (`GitHub Repository <https://github.com/BESSER-PEARL/BESSER/tree/feature/mockup_to_buml/besser/BUML/notations/mockup_to_gui/besser_integration/multiple_pages>`__)


Getting Started
----------------
Follow these steps to transform your UI mock-ups into B-UML model using BESSER:

+ Single UI Image:
    1. Navigate to the directory: 
    
       - Go to the ``besser\BUML\notations\mockup_to_gui`` directory in your project setup.
    2. Prepare your UI image:

       - Place the image in a folder at your desired path.
       - Run the model generator by providing the path to the UI mock-up as input. The generator will analyze the image and produce two Python code files: one representing the Structural model and another representing the GUI model.
       
    3. Generate the GUI model:

       - Use the following code snippet to generate the GUI model:

.. code-block:: python
    
    from besser.BUML.notations.mockup_to_gui.main_one_page import mockup_to_gui

    # Run the main function
    gui_model = mockup_to_gui(mockup_image_path="Path to your UI mock-up folder")



+ Multiple UI Images:
    1. Prepare your UI images and additional files:

       - Place the images in a folder at your desired path.
       - Prepare these files: ``Navigation image file``, ``Page order text file``, and ``Additional information text file``.
       - Refer to the :doc:`UI Mock-Up to B-UML example <../../examples/mockup_to_buml_example>` for details on creating these files.
   

    2. Generate the GUI model:

       - Use the following code snippet to generate the GUI model for multiple pages:
   

.. code-block:: python
    
    from besser.BUML.notations.mockup_to_gui.main_multiple_images import mockups_to_gui

    # Run the main function
    gui_model = mockups_to_gui(
    mockup_images_path="path to mockup images folder", 
    navigation_image_path="path to navigation image file", 
    pages_order_file_path="path to page order file",
    additional_text_file_path="path to additional text file"
    )


Access the Output
-----------------
The generator will analyze the provided UI image(s) and create: GUI model in the ``output/gui_model`` folder as a file with name ``generated_gui_model.py`` and Structural model in the ``output/buml`` folder as a file with name ``buml.py``.     

  

Example inputs and Outputs
--------------------------

Visit the :doc:`UI Mock-Up to B-UML example <../../examples/mockup_to_buml_example>` section to explore:

+	Sample input UI mockups and additional input files for multi-page cases.
+	Generated Structural Model.
+	Python-based IFML-like GUI models.
+	Integrated GUI models with navigation logic.
 