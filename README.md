# Movie Recommender System

A hybrid movie recommendation system that combines collaborative filtering and content-based approaches to provide personalized movie recommendations. Includes integration with Letterboxd for importing personal ratings.

## Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering (SVD) and content-based filtering for accurate predictions
- **Letterboxd Integration**: Import your ratings from Letterboxd exports
- **Adaptive Weighting**: Automatically adjusts recommendation strategy based on user rating history
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, Precision@K, Recall@K, NDCG@K)
- **Rich Movie Metadata**: Leverages TMDB API for detailed movie information (cast, crew, keywords)

## Architecture

### Collaborative Filtering (CF)
- Algorithm: SVD (Singular Value Decomposition)
- Best for: Finding patterns across similar users
- Handles: Personalization based on crowd wisdom

### Content-Based (CB)
- Features: Genres, cast, directors, keywords (TF-IDF weighted)
- Best for: New movies, niche preferences, cold-start scenarios
- Handles: Recommending similar movies to what you've enjoyed

### Hybrid Approach
Adaptive weighting based on user rating count:
- **New users** (<50 ratings): 30% CF, 70% CB
- **Moderate users** (50-200 ratings): 60% CF, 40% CB
- **Established users** (>200 ratings): 80% CF, 20% CB

## Project Structure

```
movie-recommender/
├── data/
│   ├── raw/                 # MovieLens 25M, Letterboxd exports (gitignored)
│   ├── processed/           # Cleaned datasets (parquet format)
│   └── external/            # TMDB API cache
├── models/                  # Trained models (.pkl files)
├── notebooks/               # Jupyter notebooks for development
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_collaborative_filtering.ipynb
│   ├── 05_content_based.ipynb
│   ├── 06_hybrid_model.ipynb
│   └── 07_evaluation.ipynb
├── src/                     # Python package
│   ├── data/                # Data loading and cleaning
│   ├── features/            # Feature engineering
│   ├── models/              # Model implementations
│   ├── matching/            # Title matching for Letterboxd
│   └── utils/               # Utilities
├── scripts/                 # Executable scripts
│   ├── download_data.py
│   ├── train_models.py
│   ├── recommend.py
│   └── letterboxd_upload.py
└── tests/                   # Unit tests
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository** (or navigate to project directory):
```bash
cd C:\Users\ololi\StudioProjects\movie-recommender
```

2. **Create virtual environment**:
```bash
python -m venv venv
```

3. **Activate virtual environment**:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install as package:
```bash
pip install -e .
```

For development (includes Jupyter, testing tools):
```bash
pip install -e ".[dev]"
```

5. **Configure TMDB API** (optional, but recommended):
   - Create account at https://www.themoviedb.org/
   - Get API key from https://www.themoviedb.org/settings/api
   - Create `config/tmdb_api_key.env`:
     ```
     TMDB_API_KEY=your_api_key_here
     ```

## Quick Start

### 1. Download MovieLens Dataset

```bash
python scripts/download_data.py
```

This downloads the MovieLens 25M dataset (~250 MB) and extracts it to `data/raw/ml-25m/`.

### 2. Explore and Clean Data

Open Jupyter notebooks:
```bash
jupyter notebook
```

Run in order:
1. `01_data_exploration.ipynb` - Understand the dataset
2. `02_data_cleaning.ipynb` - Clean and preprocess data

### 3. Train Models

```bash
python scripts/train_models.py
```

Or use notebooks:
- `03_feature_engineering.ipynb` - Prepare features
- `04_collaborative_filtering.ipynb` - Train SVD model
- `05_content_based.ipynb` - Build content-based model
- `06_hybrid_model.ipynb` - Combine into hybrid system

### 4. Get Recommendations

#### Using Letterboxd Export

1. Export your Letterboxd data:
   - Log in to Letterboxd
   - Settings → Data → Import & Export → Export Your Data
   - Download and extract `ratings.csv`

2. Get recommendations:
```bash
python scripts/recommend.py --letterboxd path/to/letterboxd_ratings.csv --top 20
```

#### Using User ID (MovieLens dataset)

```bash
python scripts/recommend.py --user-id 12345 --top 20
```

## Usage Examples

### CLI Recommendations

```bash
# Get top 10 recommendations from Letterboxd export
python scripts/recommend.py --letterboxd data/raw/letterboxd/ratings.csv --top 10

# Get recommendations with explanations
python scripts/recommend.py --letterboxd ratings.csv --top 20 --explain

# Filter by genre
python scripts/recommend.py --user-id 100 --top 15 --genre "Sci-Fi"
```

### Python API

```python
from src.models.hybrid import HybridRecommender
from src.data.loaders import load_ratings, load_movies

# Load data
ratings = load_ratings('data/processed/ratings_cleaned.parquet')
movies = load_movies('data/processed/movies_metadata.parquet')

# Initialize recommender
recommender = HybridRecommender.load('models/')

# Get recommendations
user_id = 12345
recommendations = recommender.recommend(user_id, top_n=20)

# Display results
for movie_id, score in recommendations:
    movie_title = movies.loc[movie_id, 'title']
    print(f"{movie_title}: {score:.2f}")
```

## Datasets

### MovieLens 25M
- **Source**: https://grouplens.org/datasets/movielens/25m/
- **Size**: 25 million ratings
- **Users**: 162,000
- **Movies**: 62,000
- **License**: Available for research and education

### TMDB API
- **Source**: https://www.themoviedb.org/
- **Free tier**: 40 requests / 10 seconds
- **Data**: Movie metadata, cast, crew, keywords

### Letterboxd
- **User export**: CSV format with your personal ratings
- **Format**: Date, Name, Year, Letterboxd URI, Rating

## Model Performance

Target metrics on test set:
- **RMSE**: <0.85
- **MAE**: <0.65
- **Precision@10**: >0.35
- **Recall@10**: >0.20
- **NDCG@10**: >0.40

## Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters (SVD factors, learning rate)
- Data cleaning thresholds
- Hybrid weighting strategy
- TMDB API settings
- Recommendation parameters

## Development

### Running Tests

```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=src tests/
```

### Code Formatting

```bash
black src/ scripts/ tests/
```

### Linting

```bash
flake8 src/ scripts/ tests/
```

## Notebooks Overview

1. **01_data_exploration.ipynb**: EDA on MovieLens dataset
2. **02_data_cleaning.ipynb**: Data cleaning and preprocessing
3. **03_feature_engineering.ipynb**: Create user-item matrix
4. **04_collaborative_filtering.ipynb**: Train and evaluate SVD model
5. **05_content_based.ipynb**: Build content-based recommender
6. **06_hybrid_model.ipynb**: Combine CF and CB with adaptive weighting
7. **07_evaluation.ipynb**: Comprehensive model evaluation

## Troubleshooting

### TMDB API Rate Limits
If you hit rate limits, the system automatically caches responses. Wait a few seconds and retry.

### Memory Issues
For large datasets:
- Use data subsets during development
- Increase chunk size in data loaders
- Use sparse matrices for CF

### Title Matching Issues
If Letterboxd movies aren't matching:
- Check `data/processed/unmatched_letterboxd.csv`
- Create custom mapping file
- Adjust fuzzy matching threshold in config

## Contributing

This is a learning/portfolio project. Feel free to:
- Open issues for bugs or questions
- Submit pull requests for improvements
- Fork and extend for your own use

## License

MIT License - see LICENSE file for details

## Acknowledgments

- MovieLens dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
- TMDB API for movie metadata
- scikit-surprise library for collaborative filtering
- Letterboxd for personal rating data

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with**: Python, scikit-learn, scikit-surprise, pandas, TMDB API
