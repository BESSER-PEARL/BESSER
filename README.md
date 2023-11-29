# BESSER

BESSER is a [low-modeling](https://modeling-languages.com/welcome-to-the-low-modeling-revolution/) [low-code](https://modeling-languages.com/low-code-vs-model-driven/) open-source platform. BESSER (Building bEtter Smart Software fastER) is funded thanks to an [FNR Pearl grant](https://modeling-languages.com/a-smart-low-code-platform-for-smart-software-in-luxembourg-goodbye-barcelona/) led by the [Luxembourg Institute of Science and Technology](https://www.list.lu/) with the participation of the [Snt/University of Luxembourg](https://www.uni.lu/snt-en/) and open to all your contributions!

The BESSER low-code platform is built on top of our Python-based personal interpretation of a "Universal Modeling Language" (yes, heavily inspired and a simplified version of the better known UML, the Unified Modeling Language) 

**Check out the official [documentation](https://besser.readthedocs.io/en/latest/)**

## Installation

BESSER works with Pyhton 3.9+. The pip *besser* package will be available soon, but you can also build it from source. First, clone the repository.

    $ git clone https://github.com/BESSER-PEARL/BESSER.git
    $ cd BESSER

Install *build*, then generate and install the *besser* package.

    $ pip install --upgrade build
    $ python -m build
    $ pip install dist/besser-0.1.0-py3-none-any.whl

You can check the installation of the *besser* package.

    $ pip list

## License

This project is licensed under the [MIT](https://mit-license.org/) license.