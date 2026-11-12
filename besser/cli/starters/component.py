"""A starter B-UML model: the components of a library application and the interfaces between them.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/uml_component.html
"""

from besser.BUML.metamodel.uml_component import (
    Component,
    ComponentDependency,
    ComponentModel,
    Interface,
    InterfaceProvided,
    InterfaceRequired,
    Locality,
    Subsystem,
)

# Components: the replaceable parts of the system
web_ui = Component(name="WebUI")
catalog_service = Component(name="CatalogService")
loan_service = Component(name="LoanService")
# An external component is run by someone else
payment_gateway = Component(name="PaymentGateway", locality=Locality.EXTERNAL)

# Subsystem: a component that groups other components
backend = Subsystem(name="Backend", children={catalog_service, loan_service})

# Interfaces: the contracts components offer and use
catalog_api = Interface(name="CatalogAPI")
loan_api = Interface(name="LoanAPI")

# Relationships: who provides an interface, who requires it, and plain dependencies between components
relationships = {
    InterfaceProvided(catalog_service, catalog_api, name="catalog_provides_api"),
    InterfaceProvided(loan_service, loan_api, name="loan_provides_api"),
    InterfaceRequired(web_ui, catalog_api, name="ui_requires_catalog"),
    InterfaceRequired(web_ui, loan_api, name="ui_requires_loans"),
    ComponentDependency(loan_service, payment_gateway, name="loans_use_payments"),
}

# The model: every component (the children of a subsystem too), the interfaces and the relationships.
# The besser command reads every model bound to a top-level name of this file.
library_components = ComponentModel(
    name="LibraryComponents",
    components={web_ui, backend, catalog_service, loan_service, payment_gateway},
    interfaces={catalog_api, loan_api},
    relationships=relationships,
)
