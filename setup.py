from setuptools import setup, find_packages
from lightorch import __version__, __author__, __email__
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

if __name__ == "__main__":
    setup(
        name="lightorch",
        version=__version__,
        packages=find_packages(),
        author=__author__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email=__email__,
        description="Pytorch & Lightning based framework for research and ml-pipeline automation.",
        url="https://github.com/Jorgedavyd/lightorch",
        license="MIT",
        install_requires=["lightning", "torch", "torchvision", "optuna", "tqdm"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    )
