"""
Setup script for Speech Emotion Recognition package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="speech-emotion-recognition",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning model for Speech Emotion Recognition using RAVDESS dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/speech-emotion-recognition",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/speech-emotion-recognition/issues",
        "Documentation": "https://github.com/yourusername/speech-emotion-recognition/wiki",
        "Source Code": "https://github.com/yourusername/speech-emotion-recognition",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        "api": [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
        ],
        "viz": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ser-train=train:main",
            "ser-evaluate=evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "speech emotion recognition",
        "audio processing",
        "deep learning",
        "CNN",
        "RAVDESS",
        "emotion classification",
        "affective computing",
    ],
)
