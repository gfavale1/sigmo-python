from setuptools import setup, find_packages

setup(
    name="sigmo",
    version="0.0.1",
    description="Python interface for the SIGMo subgraph isomorphism library",
    author="Giacomo Favale",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.10",
)