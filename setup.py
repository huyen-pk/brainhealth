from setuptools import setup, find_packages
import os

# Function to load requirements from requirements.txt
def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()
    
setup(
    name='brainhealth',
    version='0.1',
    packages=find_packages(where="src/brainhealth"),
    package_dir={"": "src/brainhealth"},
    install_requires=load_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    author='Huyen Pham',
    author_email='connect@huyenpk.com',
    description='A tool to assist in the diagnosis of brain\'s disease',
    url='https://github.com/huyen-pk/brainhealth',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)