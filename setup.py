"""CAPito Setup Script - Content-Aware Photo Image Text Optimizer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "transformers>=4.20.0",
        "ultralytics>=8.0.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0"
    ]

setup(
    name="capito",
    version="2.0.0",
    author="CAPito Development Team",
    description="Content-Aware Photo Image Text Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "get-caption=src.main_pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
