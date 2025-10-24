from setuptools import setup, find_packages

setup(
    name="conscious-agent",
    version="0.1.0",
    description="Conscious AI Agent with Curiosity and Alignment",
    author="Ian",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
    ],
    python_requires=">=3.9",
)