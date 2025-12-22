#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Surfacia - A comprehensive Python package for molecular structure-activity relationship (QSAR/QSPR) studies.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="surfacia",
    version="3.0.1",
    author="YumingSu",
    author_email="823808458@qq.com",
    description="A comprehensive Python package for molecular structure-activity relationship (QSAR/QSPR) studies with interactive SHAP visualization and AI-powered analysis",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/sym823808458/Surfacia",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "surfacia=surfacia.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "surfacia": [
            "data/*.csv",
            "data/*.json",
            "config/*.yaml",
            "templates/*.txt",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/sym823808458/Surfacia/issues",
        "Source": "https://github.com/sym823808458/Surfacia",
        "Documentation": "https://surfacia.readthedocs.io/",
    },
    keywords="chemistry quantum-chemistry molecular-descriptors QSAR QSPR computational-chemistry rdkit gaussian multiwfn",
)
