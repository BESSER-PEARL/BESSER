import os

from metamodel.structural.structural import DomainModel
from generators.django.auxiliary import get_constraints_for_class


class DjangoGenerator:
    def __init__(self, model, output_dir, rule_based=True, llm_based=False):
        self.model = model
        self.output_dir = output_dir
        self.rule_based = rule_based
        self.llm_based = llm_based

    def generate(self):
        self.generate_models()
        self.generate_views()
        self.generate_urls()
        self.generate_templates()

    def generate_models(self):
        # Path: MyUML\generators\django\django_models_generator.py
        if self.rule_based:
           models_generator = DjangoModelsGeneratorRuleBased(self.model, self.output_dir)
        else:
            models_generator = DjangoModelsGeneratorLLMBased(self.model, self.output_dir)
        models_generator.generate()

    def generate_views(self):
        pass
        # Path: MyUML\generators\django\django_views_generator.py
        #views_generator = DjangoViewsGenerator(self.model, self.output_dir)
        #views_generator.generate()

    def generate_urls(self):
        pass
        #Path: MyUML\generators\django\django_urls_generator.py
        #urls_generator = DjangoUrlsGenerator(self.model, self.output_dir)
        #urls_generator.generate()

    def generate_templates(self):
        pass
        # Path: MyUML\generators\django\django_templates_generator.py
        #templates_generator = DjangoTemplatesGenerator(self.model, self.output_dir)
        #templates_generator.generate()


class DjangoModelsGenerator:
    def __init__(self, model: DomainModel, output_dir:str ):
        self.model = model
        self.output_dir = output_dir

    def generate(self):
       pass

class DjangoModelsGeneratorRuleBased(DjangoModelsGenerator):
    def generate(self):
        file_name = "models.py"

        # Create the full file path by joining the folder path and file name
        file_path = os.path.join(self.output_dir, file_name)

        # Open the file in write mode
        with open(file_path, "w") as f :
            for class_to_generate in self.model.get_classes():
                class_text: str = f'''class {class_to_generate.name}(models.Model) :'''
                f.write(class_text +"\n")
                #for each attribute in the class
                for attribute in class_to_generate.attributes:
                    attribute_text: str = f'''{attribute.name} = models.CharField(max_length=100)''' #to be replaced with the actual text depending on the type
                    f.write("    " + attribute_text +"\n")
                # for each constraint in the model that has as context class_to_generate
                constraints_for_class = get_constraints_for_class(self.model, class_to_generate)
                if (len(constraints_for_class)>0):
                    f.write("\n"+"    " + "def clean(self):" + "\n")
                    for constraint in constraints_for_class:
                        validator_text = f'''if not({constraint.expression.to_string()}) : raise ValidationError("Constraint {constraint.name} violated")'''
                        f.write("        " + validator_text + "\n")
                f.write("\n")


class DjangoModelsGeneratorLLMBased(DjangoModelsGenerator):
    def generate(self):
        file_name = "models.py"

        # Create the full file path by joining the folder path and file name
        file_path = os.path.join(self.output_dir, file_name)
        # Open the file in write mode
        with open(file_path, "w") as f :
            prompt: str = "Translate the following schema to a django models.py file: \n"
            f.write(prompt)
            for element in self.model.elements:
                class_text: str = f'''A {element.name} with the following attributes:'''
                f.write(class_text +"\n")
                #for each element that it's a class
                for attribute in element.attributes:
                    attribute_text: str = f'''an {attribute.name} of type int''' #to be replaced with the actual text depending on the type
                    f.write("    " + attribute_text +"\n")
