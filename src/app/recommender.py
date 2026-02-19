"""Backend recommendation logic for the Streamlit app.

Loads pre-trained models once (cached) and exposes recommendation functions.

CB model key names (as saved by notebook 05):
  movie_feature_matrix  – sparse (n_movies, n_feats), L2-normalised
  movie_idx_lookup      – {movie_id → row in feature matrix}   ← keyed by movie_id!
  idx_to_movie_id       – {row → movie_id}
  tfidf, genre_cols, weights, rating_midpoint, evaluation
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]   # .../movie-recommender/
SRC    = ROOT / "src"
FEAT   = SRC / "data" / "features"
PROC   = SRC / "data" / "processed"
MODELS = SRC / "models"


# ── SGD-MF class – must be defined before unpickling ─────────────────────────
class SGDMatrixFactorization:
    """Stub that matches the attribute names used by the real pickled model.

    Actual attribute names (verified from saved pkl):
      user_factors, item_factors, user_biases, item_biases, global_mean
    """

    def __init__(self, n_users, n_items, n_factors=100, lr=0.005,
                 reg=0.02, n_epochs=20, random_state=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        rng = np.random.RandomState(random_state)
        self.user_factors = rng.normal(0, 0.01, (n_users, n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, n_factors))
        self.user_biases  = np.zeros(n_users)
        self.item_biases  = np.zeros(n_items)
        self.global_mean  = 0.0

    def predict(self, user_idx, item_idx):
        s = (self.global_mean
             + self.user_biases[user_idx]
             + self.item_biases[item_idx]
             + self.user_factors[user_idx] @ self.item_factors[item_idx])
        return float(np.clip(s, 0.5, 5.0))

    def predict_batch(self, user_indices, item_indices):
        s = (self.global_mean
             + self.user_biases[user_indices]
             + self.item_biases[item_indices]
             + np.sum(self.user_factors[user_indices]
                      * self.item_factors[item_indices], axis=1))
        return np.clip(s, 0.5, 5.0)


class _ModelUnpickler(pickle.Unpickler):
    """Remap SGDMatrixFactorization to our local definition at load time."""

    def find_class(self, module, name):
        if name == "SGDMatrixFactorization":
            return SGDMatrixFactorization
        return super().find_class(module, name)


# ── Model loading (cached for the whole Streamlit session) ────────────────────
@st.cache_resource(show_spinner="Loading models – first run may take ~30 s…")
def load_all() -> dict:
    """Load all pre-trained models and feature data into memory."""
    with open(FEAT / "id_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    with open(MODELS / "sgd_mf_model.pkl", "rb") as f:
        sgd = _ModelUnpickler(f).load()

    with open(MODELS / "cb_model.pkl", "rb") as f:
        cb = pickle.load(f)

    # Normalise feature matrix to unit vectors so dot-product == cosine similarity.
    # The saved matrix is NOT pre-normalised (row norms ~2–6).
    from sklearn.preprocessing import normalize as sk_normalize
    cb["movie_feature_matrix"] = sk_normalize(
        cb["movie_feature_matrix"], norm="l2"
    )

    movies_df  = pd.read_parquet(FEAT / "movie_features.parquet")
    ratings_df = pd.read_parquet(
        PROC / "ratings_cleaned.parquet",
        columns=["userId", "movieId", "rating"],
    )

    movies_by_id = movies_df.set_index("movieId")

    return dict(
        sgd=sgd,
        cb=cb,
        mappings=mappings,
        movies_df=movies_df,
        movies_by_id=movies_by_id,
        ratings_df=ratings_df,
    )


# ── CB model accessors (hide the actual key names) ────────────────────────────
def _feat_mat(cb):
    return cb["movie_feature_matrix"]           # sparse (n_movies, n_feats)

def _mid_to_row(cb):
    return cb["movie_idx_lookup"]               # {movie_id → row in feat_mat}

def _row_to_mid(cb):
    return cb["idx_to_movie_id"]                # {row → movie_id}


# ── Movie title search ────────────────────────────────────────────────────────
def search_movies(query: str, movies_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    from rapidfuzz import process, fuzz
    titles = movies_df["title"].tolist()
    hits = process.extract(query, titles, scorer=fuzz.WRatio,
                           limit=n, score_cutoff=30)
    if not hits:
        return pd.DataFrame()
    rows = [movies_df.iloc[h[2]] for h in hits]
    cols = ["movieId", "title", "genres", "year", "avg_rating", "num_ratings"]
    return pd.DataFrame(rows)[cols].reset_index(drop=True)


# ── Content-based: similar movies ────────────────────────────────────────────
def get_similar_movies(movie_id: int, models: dict, n: int = 10) -> pd.DataFrame:
    """Return n most similar movies using cosine similarity on CB feature vectors."""
    cb           = models["cb"]
    feat_mat     = _feat_mat(cb)
    mid_to_row   = _mid_to_row(cb)   # {movie_id → row}
    row_to_mid   = _row_to_mid(cb)   # {row → movie_id}
    movies_by_id = models["movies_by_id"]

    if movie_id not in mid_to_row:
        return pd.DataFrame()

    row  = mid_to_row[movie_id]
    sims = (feat_mat @ feat_mat[row].T).toarray().ravel()
    sims[row] = -1.0                          # exclude the query movie

    top_rows = np.argsort(sims)[::-1][:n]
    rows_out = []
    for r in top_rows:
        mid = row_to_mid.get(r)
        if mid is None or mid not in movies_by_id.index:
            continue
        m = movies_by_id.loc[mid]
        rows_out.append({
            "title":       m["title"],
            "genres":      m["genres"],
            "year":        m.get("year"),
            "avg_rating":  m.get("avg_rating"),
            "num_ratings": m.get("num_ratings"),
            "similarity":  round(float(sims[r]), 4),
        })
    return pd.DataFrame(rows_out)


# ── Collaborative-filtering recommendations ───────────────────────────────────
def get_cf_recs(user_id: int, models: dict, n: int = 20) -> pd.DataFrame:
    sgd          = models["sgd"]
    mappings     = models["mappings"]
    ratings_df   = models["ratings_df"]
    movies_by_id = models["movies_by_id"]

    if user_id not in mappings["user_id_map"]:
        return pd.DataFrame()

    user_idx    = mappings["user_id_map"][user_id]
    rated_ids   = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    rated_idxs  = {mappings["movie_id_map"][m] for m in rated_ids
                   if m in mappings["movie_id_map"]}

    all_midxs  = np.array(list(mappings["idx_to_movie"].keys()), dtype=np.int32)
    mask       = ~np.isin(all_midxs, list(rated_idxs))
    cand       = all_midxs[mask]
    user_arr   = np.full(len(cand), user_idx, dtype=np.int32)
    scores     = sgd.predict_batch(user_arr, cand)

    order    = np.argsort(scores)[::-1][:n]
    rows_out = []
    for midx, s in zip(cand[order], scores[order]):
        mid = mappings["idx_to_movie"].get(midx)
        if mid is None or mid not in movies_by_id.index:
            continue
        m = movies_by_id.loc[mid]
        rows_out.append({
            "title":            m["title"],
            "genres":           m["genres"],
            "year":             m.get("year"),
            "predicted_rating": round(float(s), 3),
            "avg_rating":       m.get("avg_rating"),
            "num_ratings":      m.get("num_ratings"),
        })
    return pd.DataFrame(rows_out)


# ── Build a content-based user profile ───────────────────────────────────────
def build_profile(rated_dict: dict, models: dict) -> np.ndarray | None:
    """Weighted average of feature vectors for the user's rated movies.

    rated_dict: {movie_id -> rating}
    """
    cb          = models["cb"]
    feat_mat    = _feat_mat(cb)
    mid_to_row  = _mid_to_row(cb)
    midpoint    = cb.get("rating_midpoint", 3.0)

    positions, weights = [], []
    for mid, rating in rated_dict.items():
        row = mid_to_row.get(mid)
        if row is None:
            continue
        positions.append(row)
        weights.append(float(rating) - midpoint)

    if not positions:
        return None

    w       = np.array(weights, dtype=np.float32)
    vecs    = feat_mat[positions]
    # w @ sparse_slice returns a numpy ndarray (not sparse) in scipy
    profile = np.asarray(w @ vecs).ravel()
    return profile / (np.abs(w).sum() + 1e-9)


# ── CB recommendations from a pre-built profile ───────────────────────────────
def get_cb_recs_from_profile(
    profile: np.ndarray,
    exclude_ids: set,
    models: dict,
    n: int = 20,
) -> pd.DataFrame:
    cb           = models["cb"]
    feat_mat     = _feat_mat(cb)
    mid_to_row   = _mid_to_row(cb)
    row_to_mid   = _row_to_mid(cb)
    movies_by_id = models["movies_by_id"]

    exclude_rows = {mid_to_row[m] for m in exclude_ids if m in mid_to_row}

    pv   = sp.csr_matrix(profile.reshape(1, -1))
    sims = (feat_mat @ pv.T).toarray().ravel()
    for r in exclude_rows:
        if r < len(sims):
            sims[r] = -1.0

    top_rows = np.argsort(sims)[::-1][:n]
    rows_out = []
    for r in top_rows:
        mid = row_to_mid.get(r)
        if mid is None or mid not in movies_by_id.index:
            continue
        m = movies_by_id.loc[mid]
        rows_out.append({
            "title":       m["title"],
            "genres":      m["genres"],
            "year":        m.get("year"),
            "cb_score":    round(float(sims[r]), 4),
            "avg_rating":  m.get("avg_rating"),
            "num_ratings": m.get("num_ratings"),
        })
    return pd.DataFrame(rows_out)


# ── Hybrid recommendations (adaptive alpha blend) ─────────────────────────────
def get_hybrid_recs(user_id: int, models: dict, n: int = 20) -> pd.DataFrame:
    sgd          = models["sgd"]
    mappings     = models["mappings"]
    ratings_df   = models["ratings_df"]
    movies_by_id = models["movies_by_id"]
    cb           = models["cb"]
    feat_mat     = _feat_mat(cb)
    mid_to_row   = _mid_to_row(cb)

    if user_id not in mappings["user_id_map"]:
        return pd.DataFrame()

    user_idx   = mappings["user_id_map"][user_id]
    user_rat   = ratings_df[ratings_df["userId"] == user_id]
    n_rated    = len(user_rat)
    alpha      = 0.3 if n_rated < 50 else (0.6 if n_rated < 200 else 0.8)

    rated_dict = dict(zip(user_rat["movieId"], user_rat["rating"]))
    profile    = build_profile(rated_dict, models)
    rated_idxs = {mappings["movie_id_map"][m] for m in rated_dict
                  if m in mappings["movie_id_map"]}

    all_midxs = np.array(list(mappings["idx_to_movie"].keys()), dtype=np.int32)
    mask      = ~np.isin(all_midxs, list(rated_idxs))
    cand      = all_midxs[mask]

    # CF scores
    user_arr  = np.full(len(cand), user_idx, dtype=np.int32)
    cf_scores = sgd.predict_batch(user_arr, cand)

    # CB scores (calibrated: sim → rating scale via notebook-06 calibration)
    if profile is not None:
        pv       = sp.csr_matrix(profile.reshape(1, -1))
        all_sims = (feat_mat @ pv.T).toarray().ravel()   # indexed by CB row

        cb_scores = np.full(len(cand), sgd.global_mean, dtype=np.float32)
        for i, midx in enumerate(cand):
            mid = mappings["idx_to_movie"].get(midx)
            if mid is not None:
                row = mid_to_row.get(mid)
                if row is not None and row < len(all_sims):
                    cb_scores[i] = float(0.2333 * all_sims[row] + 3.0967)
    else:
        cb_scores = cf_scores.copy()
        alpha = 1.0

    hybrid = alpha * cf_scores + (1 - alpha) * cb_scores
    order  = np.argsort(hybrid)[::-1][:n]

    rows_out = []
    for i in order:
        midx = cand[i]
        mid  = mappings["idx_to_movie"].get(midx)
        if mid is None or mid not in movies_by_id.index:
            continue
        m = movies_by_id.loc[mid]
        rows_out.append({
            "title":        m["title"],
            "genres":       m["genres"],
            "year":         m.get("year"),
            "hybrid_score": round(float(hybrid[i]), 3),
            "cf_score":     round(float(cf_scores[i]), 3),
            "cb_score":     round(float(cb_scores[i]), 3),
            "avg_rating":   m.get("avg_rating"),
            "num_ratings":  m.get("num_ratings"),
        })
    return pd.DataFrame(rows_out)
