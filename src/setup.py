from setuptools import setup, find_packages

setup(
    name="game_engine_pygame",
    version="0.1",
    package_dir={"": "src"}, 
    packages=find_packages(where="src"),
    install_requires=[
        "pygame",
        "torch",
        "gymnasium",
    ],
)