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
        'click'
    ]

setup(
    name = "coveragetools",
    version = "0.0.1",
    author = "InsilicoLab",
    description = ("Fast matching and alignment algorithm tool for real-time PCR oligo verification"),
    license = "For internal use only. Copyright © 2022. Seegene Inc., all rights reserved.",
    packages=find_packages(include=[
        'coveragetools', 
        'coveragetools.utils', 
        'coveragetools.alignment', 
        'coveragetools.amplification',
        'coveragetools.oligo',
        'coveragetools.pipeline',
        'coveragetools.report'
    ]),
    long_description=read('README.md'),
    install_requires=install_requires()
)