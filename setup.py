# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

dependencies = []
with open("environment.yml", "r") as env_file:
    is_dependencies_section = False
    for line in env_file:
        stripped_line = line.strip()
        if stripped_line == "dependencies:":
            is_dependencies_section = True
        elif (
            is_dependencies_section
            and stripped_line
            and not stripped_line.startswith("-")
        ):
            dependencies.append(stripped_line)
        elif stripped_line and not stripped_line.startswith(" "):
            is_dependencies_section = False

setup(
    name="core-analysis",
    version="0.0.1",
    author="Victor Silva dos Santos, Jérome Simon",
    author_email="victor.santos@inrs.ca, jerome.simon@inrs.ca",
    description="Core analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/groupeLIAMG/core-analysis",
    packages=find_packages(),
    install_requires=dependencies,
    setup_requires=["setuptools-git"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.7.*",
)
