# BESSER UML

Our Python-based personal interpretation of a "Universal Modeling Language" (yes, heavily inspired and a simplified version of the better known UML, the Unified Modeling Language) 

The metamodel is summarised in the following figure.

<img src="/docs/img/metamodel.jpg" alt="Metamodel" style="height: 100%; width:100%;"/>

## Installation

The requirements to install BESSER UML are:

* Python 3.10.11 or later
* *setuptools* 65.5.0 or later

You can download and install Python from [the website](https://www.python.org/downloads/). Then, you can install *setuptools* by running the following command:

    pip install setuptools

Download the latest release of the project [here](https://github.com/BESSER-PEARL/BESSER-UML/releases/latest), and install BESSER UML (buml) in your environment by executing the following command (using the wheels package):

    pip install buml-x.x.x-py3-none-any.whl

**Note:** replace *x.x.x* with version number

## BUML Documentation

* [BUML modeling](https://github.com/BESSER-PEARL/BESSER-UML/tree/main/docs/metamodel-doc.md): to model a problem, system, domain, or anything else.

* [BUML PlantUML notation](https://github.com/BESSER-PEARL/BESSER-UML/tree/main/docs/plantuml-doc.md): to use the [PlantUML](https://plantuml.com/) notation to generate a BUML model.

* [BUML Django generator](https://github.com/BESSER-PEARL/BESSER-UML/tree/main/docs/django-generator-doc.md): to generate a [Django](https://www.djangoproject.com/) application from a BUML model.