from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requires() -> List[str]:
    with open('requirements.txt', 'r') as file: 
        requirements = file.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name = 'Industrial Copper Modeling',
    version = '0.1',
    author = 'Dinakara Prabhu A',
    author_email = 'dinakaraprabhu11@gmail.com',
    packages = find_packages(),
    install_requires = get_requires()
)