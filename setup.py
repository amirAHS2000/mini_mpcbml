from setuptools import setup, find_packages

setup(
    name="mini_mpcbml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'pyyaml',
        'matplotlib',
    ],
)