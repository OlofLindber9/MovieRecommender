"""Backend recommendation logic for the Streamlit app.

Loads pre-trained models once (cached) and exposes recommendation functions.

CB model key names (as saved by notebooks 05 / 09):
  movie_feature_matrix  – sparse (n_movies, n_feats), may or may not be L2-normalised
  movie_idx_lookup      – {movie_id → row in feature matrix}   ← keyed by movie_id!
  idx_to_movie_id       – {row → movie_id}
  rating_midpoint       – float (3.0)
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # prevent OpenMP conflict (torch vs implicit)

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import streamlit as st

# ── Optional PyTorch (needed for Neural CF) ───────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]   # .../movie-recommender/
SRC    = ROOT / "src"
FEAT   = SRC / "data" / "features"
PROC   = SRC / "data" / "processed"
MODELS = SRC / "models"


# ── SGD-MF class – must be defined before unpickling ─────────────────────────
class SGDMatrixFactorization:
    """Stub that matches the attribute names used by the real pickled model."""

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


# ── Two-Tower Neural CF (must match notebook 08 architecture exactly) ─────────
if TORCH_AVAILABLE:
    class TwoTowerNCF(nn.Module):
        def __init__(self, n_users, n_items, n_genres,
                     emb_dim=64, hidden_dim=128, out_dim=32, dropout=0.2,
                     global_mean=3.5):
            super().__init__()
            self.user_emb  = nn.Embedding(n_users, emb_dim)
            self.item_emb  = nn.Embedding(n_items, emb_dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.tensor([global_mean]))
            self.user_tower = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
            )
            self.item_tower = nn.Sequential(
                nn.Linear(emb_dim + n_genres, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, user_ids, item_ids, genre_feats):
            u_repr = self.user_tower(self.user_emb(user_ids))
            i_repr = self.item_tower(
                torch.cat([self.item_emb(item_ids), genre_feats], dim=-1)
            )
            dot = (u_repr * i_repr).sum(dim=-1)
            return (dot
                    + self.user_bias(user_ids).squeeze(-1)
                    + self.item_bias(item_ids).squeeze(-1)
                    + self.global_bias)


# ── Model loading (cached for the whole Streamlit session) ────────────────────
@st.cache_resource(show_spinner="Loading models – first run may take ~30 s…")
def load_all() -> dict:
    """Load all pre-trained models and feature data into memory."""
    from sklearn.preprocessing import normalize as sk_normalize

    # id_mappings.pkl — needed for CF/hybrid/NCF; optional for CB-only mode
    mappings_path = FEAT / "id_mappings.pkl"
    if mappings_path.exists():
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
    else:
        mappings = None

    # SGD-MF (large, ~150 MB) — optional; not needed for CB-only app features
    sgd_path = MODELS / "sgd_mf_model.pkl"
    if sgd_path.exists():
        with open(sgd_path, "rb") as f:
            sgd = _ModelUnpickler(f).load()
    else:
        sgd = None

    # Load enriched CB model if available, fall back to original
    cb_path = MODELS / "cb_enriched_model.pkl"
    if not cb_path.exists():
        cb_path = MODELS / "cb_model.pkl"
    with open(cb_path, "rb") as f:
        cb = pickle.load(f)

    # Always L2-normalise — already-normalised rows are unaffected (norm=1 → no-op)
    cb["movie_feature_matrix"] = sk_normalize(
        cb["movie_feature_matrix"], norm="l2"
    )

    movies_df    = pd.read_parquet(FEAT / "movie_features.parquet")
    movies_by_id = movies_df.set_index("movieId")

    # ratings_cleaned.parquet (~314 MB) — optional; only needed for CF/hybrid recs
    ratings_path = PROC / "ratings_cleaned.parquet"
    if ratings_path.exists():
        ratings_df = pd.read_parquet(
            ratings_path, columns=["userId", "movieId", "rating"]
        )
    else:
        ratings_df = None

    # Precompute movie_idx → CB row lookup for vectorised hybrid scoring
    # (only needed when sgd and id_mappings are both available)
    midx_to_cb_row = None
    if sgd is not None and mappings is not None:
        mid_to_row = cb["movie_idx_lookup"]   # {movie_id → cb_row}
        max_midx   = max(mappings["idx_to_movie"].keys()) + 1
        midx_to_cb_row = np.full(max_midx, -1, dtype=np.int32)
        for midx, movie_id in mappings["idx_to_movie"].items():
            row = mid_to_row.get(movie_id, -1)
            midx_to_cb_row[midx] = row

    # ── Neural CF (optional) ──────────────────────────────────────────────────
    ncf              = None
    item_genre_tensor = None
    ncf_config_path  = MODELS / "neural_cf_config.yaml"
    ncf_weights_path = MODELS / "neural_cf_best.pt"

    if TORCH_AVAILABLE and mappings is not None and ncf_config_path.exists() and ncf_weights_path.exists():
        import yaml
        with open(ncf_config_path) as f:
            ncf_cfg = yaml.safe_load(f)

        genre_cols_ncf = ncf_cfg.get(
            "genre_cols",
            [c for c in movies_df.columns if c.startswith("genre_")]
        )
        n_genres = len(genre_cols_ncf)

        ncf = TwoTowerNCF(
            n_users    = ncf_cfg["n_users"],
            n_items    = ncf_cfg["n_items"],
            n_genres   = ncf_cfg["n_genres"],
            emb_dim    = ncf_cfg.get("emb_dim",    64),
            hidden_dim = ncf_cfg.get("hidden_dim", 128),
            out_dim    = ncf_cfg.get("out_dim",    32),
        )
        ncf.load_state_dict(
            torch.load(ncf_weights_path, map_location="cpu")
        )
        ncf.eval()

        # Build (n_items, n_genres) float32 tensor in movie_idx order
        movies_idx      = movies_df.set_index("movieId")
        genre_data      = np.zeros((ncf_cfg["n_items"], n_genres), dtype=np.float32)
        for movie_id, midx in mappings["movie_id_map"].items():
            if movie_id in movies_idx.index:
                genre_data[midx] = (
                    movies_idx.loc[movie_id, genre_cols_ncf].values.astype(np.float32)
                )
        item_genre_tensor = torch.from_numpy(genre_data)

    return dict(
        sgd               = sgd,
        cb                = cb,
        cb_path           = str(cb_path.name),
        mappings          = mappings,
        movies_df         = movies_df,
        movies_by_id      = movies_by_id,
        ratings_df        = ratings_df,
        midx_to_cb_row    = midx_to_cb_row,
        ncf               = ncf,
        item_genre_tensor = item_genre_tensor,
    )


def ncf_available(models: dict) -> bool:
    return models.get("ncf") is not None


# ── CB model accessors ────────────────────────────────────────────────────────
def _feat_mat(cb):
    return cb["movie_feature_matrix"]

def _mid_to_row(cb):
    return cb["movie_idx_lookup"]

def _row_to_mid(cb):
    return cb["idx_to_movie_id"]


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
    cb           = models["cb"]
    feat_mat     = _feat_mat(cb)
    mid_to_row   = _mid_to_row(cb)
    row_to_mid   = _row_to_mid(cb)
    movies_by_id = models["movies_by_id"]

    if movie_id not in mid_to_row:
        return pd.DataFrame()

    row  = mid_to_row[movie_id]
    sims = (feat_mat @ feat_mat[row].T).toarray().ravel()
    sims[row] = -1.0

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


# ── Collaborative-filtering recommendations (SGD-MF) ─────────────────────────
def get_cf_recs(user_id: int, models: dict, n: int = 20) -> pd.DataFrame:
    sgd          = models["sgd"]
    mappings     = models["mappings"]
    ratings_df   = models["ratings_df"]
    movies_by_id = models["movies_by_id"]

    if sgd is None or ratings_df is None or mappings is None:
        return pd.DataFrame()
    if user_id not in mappings["user_id_map"]:
        return pd.DataFrame()

    user_idx   = mappings["user_id_map"][user_id]
    rated_ids  = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    rated_idxs = {mappings["movie_id_map"][m] for m in rated_ids
                  if m in mappings["movie_id_map"]}

    all_midxs = np.array(list(mappings["idx_to_movie"].keys()), dtype=np.int32)
    mask      = ~np.isin(all_midxs, list(rated_idxs))
    cand      = all_midxs[mask]
    user_arr  = np.full(len(cand), user_idx, dtype=np.int32)
    scores    = sgd.predict_batch(user_arr, cand)

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


# ── Neural CF recommendations ─────────────────────────────────────────────────
def get_ncf_recs(user_id: int, models: dict, n: int = 20) -> pd.DataFrame:
    if not ncf_available(models):
        return pd.DataFrame()

    ncf               = models["ncf"]
    mappings          = models["mappings"]
    ratings_df        = models["ratings_df"]
    if ratings_df is None or mappings is None:
        return pd.DataFrame()
    movies_by_id      = models["movies_by_id"]
    item_genre_tensor = models["item_genre_tensor"]

    if user_id not in mappings["user_id_map"]:
        return pd.DataFrame()

    user_idx   = mappings["user_id_map"][user_id]
    rated_ids  = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    rated_idxs = {mappings["movie_id_map"][m] for m in rated_ids
                  if m in mappings["movie_id_map"]}

    n_items        = item_genre_tensor.shape[0]
    all_item_idxs  = torch.arange(n_items)
    u_batch        = torch.full((n_items,), user_idx, dtype=torch.long)

    with torch.no_grad():
        scores = ncf(u_batch, all_item_idxs, item_genre_tensor).numpy()

    for ridx in rated_idxs:
        if 0 <= ridx < len(scores):
            scores[ridx] = -np.inf

    order    = np.argsort(scores)[::-1][:n]
    rows_out = []
    for midx in order:
        mid = mappings["idx_to_movie"].get(int(midx))
        if mid is None or mid not in movies_by_id.index:
            continue
        m = movies_by_id.loc[mid]
        rows_out.append({
            "title":            m["title"],
            "genres":           m["genres"],
            "year":             m.get("year"),
            "predicted_rating": round(float(scores[midx]), 3),
            "avg_rating":       m.get("avg_rating"),
            "num_ratings":      m.get("num_ratings"),
        })
    return pd.DataFrame(rows_out)


# ── Build a content-based user profile ───────────────────────────────────────
def build_profile(rated_dict: dict, models: dict) -> np.ndarray | None:
    """Weighted average of CB feature vectors for the user's rated movies."""
    cb         = models["cb"]
    feat_mat   = _feat_mat(cb)
    mid_to_row = _mid_to_row(cb)
    midpoint   = cb.get("rating_midpoint", 3.0)

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

    if profile.shape[0] != feat_mat.shape[1]:
        return pd.DataFrame()   # dimension mismatch (wrong CB model)

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
    sgd            = models["sgd"]
    mappings       = models["mappings"]
    ratings_df     = models["ratings_df"]
    if sgd is None or ratings_df is None or mappings is None:
        return pd.DataFrame()
    movies_by_id   = models["movies_by_id"]
    cb             = models["cb"]
    feat_mat       = _feat_mat(cb)
    midx_to_cb_row = models["midx_to_cb_row"]
    if midx_to_cb_row is None:
        return pd.DataFrame()

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

    # CB scores — vectorised via precomputed midx→cb_row lookup
    if profile is not None:
        pv       = sp.csr_matrix(profile.reshape(1, -1))
        all_sims = (feat_mat @ pv.T).toarray().ravel()   # indexed by cb_row

        cb_rows = np.where(cand < len(midx_to_cb_row), midx_to_cb_row[cand], -1)
        valid   = (cb_rows >= 0) & (cb_rows < len(all_sims))

        cb_scores          = np.full(len(cand), sgd.global_mean, dtype=np.float32)
        cb_scores[valid]   = (0.2333 * all_sims[cb_rows[valid]] + 3.0967)
    else:
        cb_scores = cf_scores.copy()
        alpha     = 1.0

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
