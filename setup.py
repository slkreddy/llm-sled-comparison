#!/usr/bin/env python3
"""
Setup script for LLM SLED Comparison package.

Author: LaxmiKumar Reddy Sammeta
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Python implementation comparing standard LLM decoding methods with SLED optimization."

# Read requirements
def read_requirements():
    """Read requirements.txt for install_requires."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('# Optional'):
                    # Extract package name and version constraint
                    if '>=' in line:
                        requirements.append(line)
                    elif line.startswith('# Optional'):
                        break
            return requirements
    except FileNotFoundError:
        return [
            "torch>=2.0.0",
            "transformers>=4.30.0", 
            "numpy>=1.21.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "psutil>=5.8.0"
        ]

setup(
    name="llm-sled-comparison",
    version="0.1.0",
    author="LaxmiKumar Reddy Sammeta",
    author_email="slkreddysite@gmail.com",
    description="Python implementation comparing standard LLM decoding with SLED optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/slkreddy/llm-sled-comparison",
    packages=find_packages(),
    py_modules=["sled_comparison"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "plotting": [
            "plotly>=5.0.0",
        ],
        "serving": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0", 
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "plotly>=5.0.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sled-comparison=sled_comparison:main",
        ],
    },
    keywords="llm, sled, decoding, optimization, pytorch, transformers, memory-efficiency",
    project_urls={
        "Bug Reports": "https://github.com/slkreddy/llm-sled-comparison/issues",
        "Source": "https://github.com/slkreddy/llm-sled-comparison",
        "Documentation": "https://github.com/slkreddy/llm-sled-comparison#readme",
    },
}
