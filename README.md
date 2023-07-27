# BESSER UML

Our Python-based personal interpretation of a "Universal Modeling Language" (yes, heavily inspired and a simplified version of the better known UML, the Unified Modeling Language) 

The metamodel is summarised in the following figure.

<img src="/docs/img/metamodel.jpg" alt="Metamodel" style="height: 100%; width:100%;"/>

## Installation Guide

The requirements to install BESSER UML are:

* Python 3.10.11 or later
* *setuptools* 65.5.0 or later

You can download and install Python from [the website](https://www.python.org/downloads/). Then, you can install *setuptools* by running the following command:

    pip install setuptools

Download the latest release of the project [here](https://github.com/BESSER-PEARL/BESSER-UML/releases/latest), and install BESSER UML (buml) in your environment by executing the following command (using the wheels package):

    pip install buml-x.x.x-py3-none-any.whl

**Note:** replace *x.x.x* with version number

## Getting Started

...

    from metamodel.structual import DomainModel, Class, Constraint