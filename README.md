<div align="center">
  <img src="./docs/source/_static/besser_logo_light.png" alt="BESSER platform" width="500"/>
</div>

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=gold)](https://pypi.org/project/besser/)
[![PyPI version](https://img.shields.io/pypi/v/besser?logo=pypi&logoColor=white)](https://pypi.org/project/besser/)
[![PyPI - Downloads](https://static.pepy.tech/badge/besser)](https://pypi.org/project/besser/)
[![Documentation Status](https://readthedocs.org/projects/besser/badge/?version=latest)](https://besser.readthedocs.io/en/latest/)
[![PyPI - License](https://img.shields.io/pypi/l/besser)](https://opensource.org/license/MIT)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?logo=Linkedin&logoColor=white)](https://www.linkedin.com/company/besser-pearl)
[![GitHub Repo stars](https://img.shields.io/github/stars/besser-pearl/besser?style=social)](https://star-history.com/#besser-pearl/besser)

BESSER is a [low-modeling](https://modeling-languages.com/welcome-to-the-low-modeling-revolution/) [low-code](https://lowcode-book.com/) open-source platform. BESSER (Building bEtter Smart Software fastER) is funded thanks to an [FNR Pearl grant](https://modeling-languages.com/a-smart-low-code-platform-for-smart-software-in-luxembourg-goodbye-barcelona/) led by the [Luxembourg Institute of Science and Technology](https://www.list.lu/) with the participation of the [Snt/University of Luxembourg](https://www.uni.lu/snt-en/) and open to all your contributions!

The BESSER low-code platform is built on top of [B-UML](https://besser.readthedocs.io/en/latest/buml_language.html) our Python-based personal interpretation of a "Universal Modeling Language" (yes, heavily inspired and a simplified version of the better known UML, the Unified Modeling Language). 
With B-UML you can specify your software application and then use any of the [code-generators available](https://besser.readthedocs.io/en/latest/generators.html) to translate your model into executable code suitable for various applications, such as Django web apps or database structures compatible with SQLAlchemy.

**Check out the [BESSER Web Modeling Editor online](https://editor.besser-pearl.org/)**
![BESSER Web Modeling Editor Demo](./docs/source/img/besser_new.gif)

**Check out the official [documentation](https://besser.readthedocs.io/en/latest/)**

## Basic Installation

BESSER works with Python 3.10+. We recommend creating a virtual environment (e.g. [venv](https://docs.python.org/3/tutorial/venv.html), [conda](https://docs.conda.io/en/latest/)).

The latest stable version of BESSER is available in the Python Package Index (PyPi) and can be installed using

    $ pip install besser

BESSER can be used with any of the popular IDEs for Python development such as [VScode](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/), [Sublime Text](https://www.sublimetext.com/), etc.

## Running BESSER Locally

If you are interested in developing new code generators or designing BESSER extensions, you can download and modify the full codebase, including tests and examples.

### Step 1: Clone the repository

    $ git clone https://github.com/BESSER-PEARL/BESSER.git
    $ cd BESSER

### Step 2: Create a virtual environment

Run the setup script to create a virtual environment (if not already created), install the requirements, and configure the ``PYTHONPATH``. This ensures compatibility with IDEs (like VSCode) that may not automatically set the ``PYTHONPATH`` for recognizing *besser* as an importable module.

    $ python setup_environment.py

**Note:** Each time you start your IDE, run the `setup_environment.py` script to ensure the environment is properly configured.

### Step 3: Run an example

To verify the setup, you can run a basic example.

    $ cd tests/BUML/metamodel/structural/library
    $ python library.py

## Examples
If you want to try examples, check out the [BESSER-examples](https://github.com/BESSER-PEARL/BESSER-examples) repository!

## Contributing

We encourage contributions from the community and any comment is welcome!

If you are interested in contributing to this project, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## How to cite BESSER

This repository has the CITATION.cff file, which activates the "Cite this repository" button in the About section (right side of the repository). The citation is in APA and BibTex format.

## Code of Conduct

At BESSER, our commitment is centered on establishing and maintaining development environments that are welcoming, inclusive, safe and free from all forms of harassment. All participants are expected to voluntarily respect and support our [Code of Conduct](CODE_OF_CONDUCT.md).

## Governance

The development of this project follows the governance rules described in the [GOVERNANCE.md](GOVERNANCE.md) document.

## Contact
You can reach us at: [info@besser-pearl.org](mailto:info@besser-pearl-org)

Website: https://besser-pearl.org

## License

This project is licensed under the [MIT](https://mit-license.org/) license.
