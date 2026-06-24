from pathlib import Path
from setuptools import setup, find_packages

# ==========================================================
# Read README
# ==========================================================

this_directory = Path(__file__).parent

readme_path = this_directory / "README.md"

long_description = ""

if readme_path.exists():
    long_description = readme_path.read_text(
        encoding="utf-8"
    )

# ==========================================================
# Setup
# ==========================================================

setup(
    name="causal_multimodal_dti",

    version="1.0.0",

    author="Roshan Kotkondawar, Kunal Nicose",

    author_email="kunal.nicose@outlook.com",

    description=(
        "Causal Multi-Modal Transformer Framework "
        "with Uncertainty-Aware Explainability "
        "for Drug–Target Interaction Prediction"
    ),

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/yourusername/causal-multimodal-dti",

    license="MIT",

    packages=find_packages(),

    include_package_data=True,

    python_requires=">=3.10",

    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "torchaudio>=2.3.0",

        "torch-geometric>=2.5.0",

        "fair-esm>=2.0.0",

        "transformers>=4.41.0",

        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.13.0",

        "scikit-learn>=1.5.0",

        "rdkit>=2023.9.6",

        "networkx>=3.3",

        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "plotly>=5.22.0",

        "captum>=0.7.0",
        "shap>=0.45.0",

        "umap-learn>=0.5.6",

        "biopython>=1.84",

        "PyYAML>=6.0",

        "omegaconf>=2.3.0",

        "tqdm>=4.66.0",

        "rich>=13.7.0",

        "tensorboard>=2.17.0",

        "wandb>=0.17.0",

        "joblib>=1.4.0",

        "requests>=2.32.0",

        "vina>=1.2.5",

        "einops>=0.8.0",

        "accelerate>=0.31.0"
    ],

    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "jupyter"
        ]
    },

    classifiers=[
        "Development Status :: 4 - Beta",

        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",

        "Programming Language :: Python :: 3.10",

        "Programming Language :: Python :: 3.11",

        "Topic :: Scientific/Engineering",

        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],

    keywords=[
        "drug-target interaction",
        "DTI",
        "drug discovery",
        "transformer",
        "graph neural network",
        "graph transformer",
        "protein language model",
        "esm2",
        "bioinformatics",
        "explainable ai",
        "causal ai",
        "uncertainty quantification",
        "deep learning",
        "computational biology"
    ],

    project_urls={
        "Source":
            "https://github.com/yourusername/causal-multimodal-dti",

        "Bug Tracker":
            "https://github.com/yourusername/causal-multimodal-dti/issues",

        "Documentation":
            "https://github.com/yourusername/causal-multimodal-dti"
    },

    zip_safe=False
)