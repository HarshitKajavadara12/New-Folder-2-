"""
GIGA SYSTEM - Setup Configuration
Greek Intelligence for Global Analysis
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="giga-system",
    version="1.0.0",
    author="GIGA Contributors",
    author_email="giga@quantfinance.dev",
    description="Greek Intelligence for Global Analysis - Quantitative Finance Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giga-system/giga",
    project_urls={
        "Bug Tracker": "https://github.com/giga-system/giga/issues",
        "Documentation": "https://giga-system.readthedocs.io",
        "Source": "https://github.com/giga-system/giga",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: R",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "numba>=0.58.0",
        "sympy>=1.12",
        "polars>=0.19.0",
        "duckdb>=0.9.0",
        "pyarrow>=14.0.0",
        "QuantLib>=1.32",
        "qiskit>=0.45.0",
        "qiskit-aer>=0.13.0",
        "vectorbt>=0.26.0",
        "rpy2>=3.5.0",
        "streamlit>=1.29.0",
        "plotly>=5.18.0",
        "toml>=0.10.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
        "quantum": [
            "qiskit-optimization>=0.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "giga=visualization.app:main",
            "giga-setup=scripts.setup_environment:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
