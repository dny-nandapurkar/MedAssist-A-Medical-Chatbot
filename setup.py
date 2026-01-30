from setuptools import find_packages, setup
setup(
    name = 'Generative AI Project',
    version = '0.0.0',
    author = 'Dnyanda',
    author_email = 'dny.nandapurkar@gmail.com',
    packages = find_packages(),  # will find __init__.py file and when found src folder will be considered as local package
    install_requires = []
)