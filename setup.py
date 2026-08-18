from setuptools import setup, find_packages

setup(
    name="spec-wave",
    version="0.1.0",
    description="Holistic Spectral Wave Language Synthesis & Parallel Vocoding Framework",
    author="Mario Raúl Carbonell Martínez",
    url="https://github.com/mcarbonell/spec-wave",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.10",
)
