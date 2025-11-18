from setuptools import setup, find_packages

setup(
    name="high_tab_priors",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "pyyaml",
        "tqdm",
    ],
)