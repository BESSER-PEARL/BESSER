from besser.BUML.metamodel.structural import *
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.notations.structuralDrawioXML import save_buml_to_file

model = plantuml_to_buml(plantUML_model_path="test.plantuml")


save_buml_to_file(model=model, file_name="test.py")

#for cl in model.get_classes():
#    print(cl.name + "---" + str(cl.is_abstract))
#    for met in cl.methods:
#        print(met.name)
#        for param in met.parameters:
#            print("Param: " + param.name + ", type: " + str(type(param.type)) + ", default_value= " + str(param.default_value))

for asso in model.associations:
    print(asso.name)
    for end in asso.ends:
        print(end.name)
        #print(end.type.name)
        print(end.is_composite)
        print(end.multiplicity)

for gen in model.generalizations:
    print(gen.general.name)
    print(gen.specific.name)