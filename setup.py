from setuptools import setup, find_packages

setup(
    name="movie-recommender",
    version="0.1.0",
    description="Hybrid movie recommender system with Letterboxd integration",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "scikit-learn>=1.3.0",
        "scikit-surprise>=1.1.3",
        "scipy>=1.11.0",
        "rapidfuzz>=3.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "ipykernel>=6.25.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
        "dashboard": [
            "streamlit>=1.25.0",
        ],
        # Neural CF (install PyTorch separately — see notebook 08)
        # Windows: conda install -c conda-forge implicit  (for notebook 07 ALS variant)
        "ml": [
            "torch",   # CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
        ],
    },
    python_requires=">=3.8",
)
