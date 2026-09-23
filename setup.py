"""Setup script for qu_alg_qu_sim_tensor_networks."""

from setuptools import setup, find_packages

VERSION_TEMPLATE = '''
"""Automatically generated."""
__version__ = {version!r}
'''.lstrip()

setup(
    name="qu_alg_qu_sim_tensor_networks",
    description="Quantum algorithms and quantum simulation via tensor networks (MPS/MPO, DMRG, TDVP).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Younes Javanmard",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "networkx>=3.0",
        "qutip>=4.7",
        "matplotlib>=3.6",
    ],
    extras_require={
        # GPU acceleration (optional)
        "gpu": [
            "cupy-cuda12x",
        ],
        "jax": [
            "jax[cuda12_pip]",
        ],
        # Quantum chemistry integration
        "chem": [
            "openfermion>=1.5",
            "pyscf>=2.0",
        ],
        # Development and testing
        "dev": [
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "pylint",
            "pydocstyle",
            "pre-commit",
            "mypy",
        ],
    },
    use_scm_version={
        "write_to": "src/qu_alg_qu_sim_tensor_networks/version.py",
        "write_to_template": VERSION_TEMPLATE,
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
