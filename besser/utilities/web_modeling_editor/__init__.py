"""BESSER Web Modeling Editor: the FastAPI backend and the agent simulator.

A regular package (not a namespace package) so that ``packages=find:`` in
``setup.cfg`` ships the backend in the wheel and ``besser-wme-backend`` works
from a PyPI install. The frontend is a separate git submodule and is not
packaged.
"""
