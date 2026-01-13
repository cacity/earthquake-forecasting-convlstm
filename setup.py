"""
Setup script for earthquake forecasting package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="eqgrid",
    version="1.0.0",
    description="Grid-based earthquake forecasting with ConvLSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm/issues",
        "Documentation": "https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm",
        "Source Code": "https://github.com/YOUR_USERNAME/earthquake-forecasting-convlstm",
        "Paper": "https://doi.org/10.1109/ACCESS.2026.XXXXXXX",
    },
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
        ],
        "geo": [
            "cartopy>=0.22.0",
            "obspy>=1.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="earthquake forecasting deep-learning convlstm spatiotemporal seismology",
)
