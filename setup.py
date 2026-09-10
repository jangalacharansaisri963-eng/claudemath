from setuptools import setup, find_packages

setup(
    name="claudemath",
    version="1.0.0",
    description="A comprehensive, advanced pure-Python mathematics library featuring 1,400+ mathematical algorithms across 20 core domains.",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="claudemath contributors",
    packages=find_packages(include=["claudemath", "claudemath.*"]),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
