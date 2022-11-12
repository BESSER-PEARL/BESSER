from textx import metamodel_from_file

myuml_meta = metamodel_from_file('../../MyUML/notations/textx/myuml.tx')

hello_myulm_model = myuml_meta.model_from_file('hello_world.myuml')

print("Greeting", ", ".join([to_greet.name
                            for to_greet in hello_myulm_model.to_greet]))