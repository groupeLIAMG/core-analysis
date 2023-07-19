# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

GITHUB_REQUIREMENT = "{name} @ git+https://github.com/{author}/{name}.git"
REQUIREMENTS = [
    "segmentation-models",
    GITHUB_REQUIREMENT.format(
        author="lucasb-eyer",
        name="pydensecrf",
    ),
    "imgaug",
    "proplot",
    "pycocotools",
    "pillow",
    "tensorflow==2.13.0",
    "opencv-python",
    "numpy",
]

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
    install_requires=REQUIREMENTS,
    setup_requires=["setuptools-git"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
