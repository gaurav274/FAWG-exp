from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="srtml_exp",
    version="0.0.1",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.6, <3.8",
    entry_points={
        "console_scripts": [
            "cleanmr=srtml_exp.scripts.model_repository:cleanmr",
            "lsmr=srtml_exp.scripts.model_repository:lsmr",
            "plsmr=srtml_exp.scripts.model_repository:plsmr",
            "start_srtml=srtml_exp.scripts.model_repository:start_srtml",
            "prepoc_profile=srtml_exp.scripts." "experiments:prepoc_profile",
            "prepoc_populate=srtml_exp.scripts." "experiments:prepoc_populate",
            "prepoc_configure=srtml_exp.scripts."
            "experiments:prepoc_configure",
            "prepoc_provision=srtml_exp.scripts."
            "experiments:prepoc_provision",
        ],
    },
    install_requires=[],
)