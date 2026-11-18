"""A starter B-UML model: the feature model of a product line of e-readers.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/feature_model.html
"""

from besser.BUML.metamodel.feature_model import (
    Feature,
    FeatureModel,
    FeatureValue,
)

# Features: what a product can have. A feature's value is an attribute to fill in when configuring a product.
e_reader = Feature(name="EReader")
screen = Feature(name="Screen")
e_ink = Feature(name="EInk")
lcd = Feature(name="LCD")
storage = Feature(name="Storage", value=FeatureValue(t="int", values=[8, 16, 32]))
connectivity = Feature(name="Connectivity")
wifi = Feature(name="WiFi")
cellular = Feature(name="Cellular")
front_light = Feature(name="FrontLight")
# min and max: how many instances of the feature a product may have (1 and 1 unless given)
user_profile = Feature(name="UserProfile", min=1, max=5)

# The tree: each call adds a group of children to a feature and returns that feature
screen.alternative([e_ink, lcd])          # alternative: exactly one of the children
connectivity.or_([wifi, cellular])        # or: at least one of the children
e_reader.mandatory(screen)                # mandatory: every product has the child
e_reader.mandatory(storage)
e_reader.mandatory(user_profile)
e_reader.optional(connectivity)           # optional: a product may have the child
e_reader.optional(front_light)

# The model, with the root of the tree. The besser command reads every model bound to a top-level name of this file.
e_reader_features = FeatureModel(name="EReaderFeatures").root(e_reader)
