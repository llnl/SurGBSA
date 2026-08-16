"""Setup script for SurGBSA package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="sur_gbsa",
    version="0.1.0",
    author="Derek Jones",
    author_email="djones@llnl.gov",
    description="Learning representations from molecular dynamics simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llnl/SurGBSA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ipython>=8.0.0",
            "ipdb>=0.13.0",
            "jupyter>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
