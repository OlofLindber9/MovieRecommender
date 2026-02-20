# ReelRecs — Movie Recommender System

A full end-to-end movie recommendation system built on the **MovieLens 25M** dataset.
Trained five distinct models (collaborative filtering, content-based, hybrid, ranking, and neural),
then packaged them into an interactive Streamlit app.

**Live demo:** https://oloflindber9-reelrecs.hf.space/

---

## What it does

The app has three entry points for getting recommendations:

- **Movie Search** — type any movie title and get the most similar films, ranked by content similarity (genres, tags, TMDB metadata)
- **For You** — enter a MovieLens user ID and get personalised picks from the hybrid CF+CB model
- **Letterboxd Import** — upload your Letterboxd `ratings.csv` export and get recommendations built from your personal watch history

---

## Dataset

**MovieLens 25M** — a standard benchmark dataset from the GroupLens research lab.

| | Raw | After cleaning |
|---|---|---|
| Ratings | 25,000,095 | 24,914,810 |
| Users | 162,541 | 162,112 |
| Movies | 59,047 | 32,424 |
| Sparsity | 99.74% | 99.53% |

Cleaning removed movies with fewer than 5 ratings, users with fewer than 20 ratings, and 403 users who assigned the same rating to >95% of movies.

Rating scale: 0.5 – 5.0 (half-star). Mean rating: **3.53**.

---

## Models

Five models were trained and evaluated, in increasing complexity.

### 1. Collaborative Filtering (SGD Matrix Factorisation)

Factorises the 162k × 32k user-item matrix into latent user and item embeddings, trained by stochastic gradient descent with L2 regularisation. Captures the "users like you also liked…" signal.

| Hyperparameter | Value |
|---|---|
| Latent factors | 100 |
| Epochs | 20 |
| Learning rate | 0.005 |
| Regularisation | 0.02 |

Baselines compared: mean+bias baseline, SVD, item-based KNN, user-based KNN.

| Model | RMSE | MAE |
|---|---|---|
| Baseline (mean + bias) | 1.0060 | 0.7672 |
| SVD | 1.0042 | 0.7654 |
| KNN Item-Based | 1.0628 | 0.8240 |
| KNN User-Based | 1.0700 | 0.8316 |
| **SGD-MF** | **1.0019** | **0.7678** |

Evaluation used a temporal 80/20 split: models were trained on the oldest 80% of ratings and tested on the most recent 20%.

### 2. Content-Based Filtering

Builds a 1,021-dimensional sparse feature vector per movie and computes cosine similarity between movies. Features:

- **20 genre flags** (multi-hot encoded, weight ×2)
- **1,000 TF-IDF tag features** from user-generated tags (weight ×1)
- **Normalised release year** (weight ×0.5)

User preferences are represented as a weighted average of the movie vectors they have rated (weight = rating − 3.0 midpoint, so liked movies pull the profile positive and disliked movies pull it negative).

RMSE on a 100k-sample from the test set: **1.1027** — worse than CF for rating prediction, but it works for cold-start movies and new users where CF has nothing to work with.

### 3. Adaptive Hybrid (CF + CB)

Blends the SGD-MF score and the CB-calibrated score with a weight α that scales with how much history the user has:

| History | CF weight (α) | CB weight (1−α) |
|---|---|---|
| < 50 ratings (new user) | 30% | 70% |
| 50–200 ratings | 60% | 40% |
| > 200 ratings (power user) | 80% | 20% |

The CB score is first calibrated with a linear regression fit on training data (`rating = 0.233 × similarity + 3.097`) so both models output on the same scale before blending.

Hybrid RMSE on 50k test sample: **1.0716** — sits between pure CF and pure CB, with the key advantage of handling new users gracefully.

### 4. Ranking Models (BPR & ALS)

Unlike the rating-prediction models above, these treat recommendations as a **ranking problem** on implicit feedback (was a movie rated at all?).

**BPR (Bayesian Personalised Ranking)** — samples (user, positive item, negative item) triplets and optimises a pairwise ranking loss. Implemented from scratch in NumPy with a vectorised sampler.

**ALS (Alternating Least Squares)** — least-squares CF on confidence-weighted implicit data (`c = 1 + 40 × rating`), using the `implicit` library.

Evaluation on 2,000 sampled users:

| Model | P@10 | nDCG@10 |
|---|---|---|
| BPR | **0.042** | **0.042** |
| ALS | 0.029 | 0.030 |

BPR outperforms ALS across all cut-offs. Both are weaker than the rating-prediction models on precision, but optimise directly for ranking order rather than score accuracy.

### 5. Two-Tower Neural CF

A deep learning model where user and item representations are learned by separate MLP towers, then combined via dot product.

**Architecture:**

```
User Tower:  Embedding(user_id, 64) → Linear(64→128) → ReLU → Linear(128→32)
Item Tower:  Embedding(item_id, 64) || genre_vector(20) → Linear(84→128) → ReLU → Linear(128→32)
Score:       dot(user_repr, item_repr) + user_bias + item_bias + global_bias
```

Total parameters: **12.67M**

| Setting | Value |
|---|---|
| Batch size | 4,096 |
| Optimiser | Adam (lr=1e-3, weight decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Epochs | 10 |
| Loss | MSE |

**Test RMSE: 0.9935** — the best of all models. Beats SGD-MF by ~0.009 RMSE points.

---

## TMDB Feature Enrichment

The CB model was also enriched with data from The Movie Database (TMDB) API for the 5,000 most-rated movies. For each movie the API provides:

- Plot overview (free text)
- Director name (weighted ×3 in TF-IDF)
- Top 5 cast members (weighted ×2)
- TMDB keywords

These are combined with the existing user tags and processed through a **3,000-feature TF-IDF** (up from 1,000), producing a 3,020-dimensional enriched feature matrix. This significantly improves similarity quality — e.g. *Toy Story* → *A Bug's Life*, *Monsters, Inc.* (vs. broad genre matches without TMDB).

The enriched model is saved as `cb_enriched_model.pkl` and used by the app when available.

---

## Tech stack

Python · NumPy · SciPy · pandas · scikit-learn · PyTorch · Streamlit · rapidfuzz · implicit · TMDB API · Hugging Face Spaces

---

## Data

- **MovieLens 25M** — F. Maxwell Harper and Joseph A. Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 5, 4 (2015). Available for non-commercial research use.
- **TMDB** — movie metadata via the TMDB API (free tier, 40 req/10s)
