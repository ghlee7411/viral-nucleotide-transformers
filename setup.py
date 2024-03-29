import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def install_requires():
    return [
        'biopython>=1.79',
        'pandas',
        'click',
        'transformers',
        'datasets',
        'tokenizers'
    ]

setup(
    name = "vinucmer",
    version = "dev",
    author = "karlo.lee",
    description = ("Viral Nucleotide Transformers"),
    license = "MIT",
    packages=find_packages(include=[
        'vinucmer',
        'vinucmer.*'
    ]),
    long_description=read('README.md'),
    install_requires=install_requires()
)