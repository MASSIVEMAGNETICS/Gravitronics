"""Setup configuration for the Gravitronics package."""

from setuptools import setup, find_packages

setup(
    name="gravitronics",
    version="1.0.0",
    description="Lightweight Gravitational Transformer (LGT) framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gravitronics Contributors",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
