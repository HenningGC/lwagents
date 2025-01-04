from setuptools import setup, find_packages

setup(
    name="lwagents",
    version="0.1.0",
    author="Henning Gruhl",
    author_email="henning@gruhl.me",
    description="A lightweight library for building graph-driven AI agents with tool integration.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HenningGC/lwagents",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.10.0",
        "dotenv",
        "openai",
        # Add other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
