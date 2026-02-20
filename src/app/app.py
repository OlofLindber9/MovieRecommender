"""Movie Recommender — Streamlit Dashboard

Run from the project root:
    streamlit run src/app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ReelRecs · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from recommender import (
    load_all,
    search_movies,
    get_similar_movies,
    get_ncf_recs,
    get_hybrid_recs,
    build_profile,
    get_cb_recs_from_profile,
    ncf_available,
)
from letterboxd import parse_letterboxd_csv, match_to_movielens


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Animated gradient hero title */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f5a623, #e85d04, #c1121f, #9d4edd);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
    margin-bottom: 0;
}
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
    color: #e0e0e0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(15, 52, 96, 0.5);
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #f5a623; }
.metric-label { font-size: 0.75rem; color: #a0aec0; margin-top: 2px; }

/* Movie rating chip */
.rating-chip {
    display: inline-block;
    background: #f5a623;
    color: #1a1a2e;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 700;
}

/* Pulse animation for "computing" state */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.computing { animation: pulse 1.5s ease-in-out infinite; color: #f5a623; }

/* Section header with accent bar */
.section-header {
    border-left: 4px solid #f5a623;
    padding-left: 0.75rem;
    margin: 1rem 0 0.5rem 0;
    font-weight: 700;
    font-size: 1.1rem;
}

/* Fade-in for results */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.4s ease forwards; }

/* Tag badges for genres */
.genre-tag {
    display: inline-block;
    background: rgba(157, 78, 221, 0.2);
    border: 1px solid rgba(157, 78, 221, 0.4);
    color: #c084fc;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.72rem;
    margin: 1px;
}

/* Sidebar model badge */
.model-badge {
    background: linear-gradient(90deg, #f5a623, #e85d04);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}

/* Dataframe header colour override */
[data-testid="stDataFrame"] th {
    background-color: #1a1a2e !important;
    color: #f5a623 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
try:
    models = load_all()
    models_ok = True
except Exception as exc:
    st.error(f"**Could not load models:** {exc}\n\nRun notebooks 01–06 first.")
    models_ok = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🎬 ReelRecs</div>', unsafe_allow_html=True)
    st.caption("AI-powered movie recommendations")
    st.divider()

    st.markdown("**Active models**")
    cb_label = "Enriched CB (TMDB)" if models_ok and "enriched" in models.get("cb_path","") else "Content-Based"
    _sgd_ok = models_ok and models.get("sgd") is not None
    st.markdown(f"""
{"- <span class='model-badge'>SGD-MF</span> RMSE 1.002" if _sgd_ok else ""}
{"- <span class='model-badge'>Hybrid</span> CF + CB blend" if _sgd_ok else ""}
- <span class="model-badge">{cb_label}</span> TF-IDF + genres
{"- <span class='model-badge'>Neural CF</span> RMSE 0.994 ✨" if models_ok and ncf_available(models) else ""}
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**Dataset**")
    st.markdown("""
- 📊 25 M ratings
- 👥 162 k users
- 🎬 32 k movies
    """)
    st.divider()
    n_recs = st.slider("Recommendations to show", 5, 50, 20, step=5)

if not models_ok:
    st.stop()

movies_df  = models["movies_df"]
ratings_df = models["ratings_df"]


# ── Session state for "For You" taste profile ─────────────────────────────────
if "my_ratings" not in st.session_state:
    st.session_state.my_ratings = {}   # {movie_id: {"title": str, "rating": float}}


# ── Helper: format results DataFrame ─────────────────────────────────────────
def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        out["year"] = out["year"].apply(lambda x: int(x) if pd.notna(x) else "")
    for col in ("avg_rating", "cb_score", "hybrid_score", "predicted_rating", "similarity"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "num_ratings" in out.columns:
        out["num_ratings"] = out["num_ratings"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
    out.index = range(1, len(out) + 1)
    return out


_RENAME = {
    "title":            "Title",
    "genres":           "Genres",
    "year":             "Year",
    "avg_rating":       "Avg ★",
    "num_ratings":      "# Ratings",
    "similarity":       "Similarity",
    "predicted_rating": "Predicted ★",
    "hybrid_score":     "Score",
    "cf_score":         "CF Score",
    "cb_score":         "CB Score",
    "ml_title":         "ML Title",
    "match_score":      "Match %",
    "rating":           "Your ★",
}


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Movie Search",
    "🎯 For You",
    "📋 Letterboxd",
    "ℹ️ How It Works",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · Movie Search / Similar Movies
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Find Similar Movies</div>', unsafe_allow_html=True)
    st.caption("Search for any movie and we'll find the most similar ones using TF-IDF tag embeddings + genre features.")

    query = st.text_input("Movie title", placeholder="e.g. Inception, Parasite, The Godfather…", key="search_q")

    if query:
        matches = search_movies(query, movies_df, n=10)

        if matches.empty:
            st.warning("No movies found — try a different spelling.")
        else:
            left, right = st.columns([1, 2])

            with left:
                st.markdown("**Matching titles**")
                choice = st.radio("Select", matches["title"].tolist(), label_visibility="collapsed")

            sel_row  = matches[matches["title"] == choice].iloc[0]
            movie_id = int(sel_row["movieId"])

            with right:
                st.markdown(f"### {choice}")
                yr = sel_row.get("year")
                c1, c2, c3 = st.columns(3)
                c1.metric("Year",       int(yr) if pd.notna(yr) else "N/A")
                c2.metric("Avg Rating", f"{sel_row.get('avg_rating', 0):.2f} ★")
                c3.metric("# Ratings",  f"{int(sel_row.get('num_ratings', 0)):,}")
                # Genre tags
                genres_html = " ".join(
                    f'<span class="genre-tag">{g.strip()}</span>'
                    for g in str(sel_row.get("genres", "")).split("|")
                )
                st.markdown(genres_html, unsafe_allow_html=True)

            st.divider()
            st.markdown(f'<div class="section-header">Movies similar to <em>{choice}</em></div>', unsafe_allow_html=True)

            with st.spinner("Computing similarity…"):
                similar = get_similar_movies(movie_id, models, n=n_recs)

            if similar.empty:
                st.info("This movie isn't in the content feature matrix.")
            else:
                st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                st.dataframe(_fmt(similar).rename(columns=_RENAME), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · For You — build taste profile from scratch
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Build Your Taste Profile</div>', unsafe_allow_html=True)
    st.caption(
        "Tell us what you've watched and how much you liked it. "
        "We'll recommend movies tailored to your taste."
    )

    col_input, col_list = st.columns([3, 2])

    # ── Left: search + add ───────────────────────────────────────────────────
    with col_input:
        st.markdown("**Search and add movies you've seen:**")
        query2 = st.text_input("Movie title", placeholder="e.g. The Dark Knight…", key="foryou_q")

        if query2:
            hits = search_movies(query2, movies_df, n=8)
            if hits.empty:
                st.warning("No results — try a different spelling.")
            else:
                selected_title = st.radio(
                    "Pick a result", hits["title"].tolist(), label_visibility="collapsed", key="foryou_radio"
                )
                sel = hits[hits["title"] == selected_title].iloc[0]
                mid = int(sel["movieId"])

                yr = sel.get("year")
                st.caption(
                    f"{'(' + str(int(yr)) + ')' if pd.notna(yr) else ''} "
                    f"· {sel.get('genres','')}"
                    f" · ⭐ {sel.get('avg_rating', 0):.2f} avg"
                )

                rating_val = st.slider(
                    "Your rating", 1.0, 5.0, 4.0, step=0.5,
                    format="%.1f ★", key="foryou_rating"
                )

                if st.button("➕ Add to my list", type="primary", key="foryou_add"):
                    st.session_state.my_ratings[mid] = {
                        "title":  selected_title,
                        "rating": rating_val,
                    }
                    st.success(f"Added **{selected_title}** ({rating_val:.1f} ★)")

    # ── Right: current list ──────────────────────────────────────────────────
    with col_list:
        n_in_list = len(st.session_state.my_ratings)
        st.markdown(f"**Your list** ({n_in_list} movie{'s' if n_in_list != 1 else ''})")

        if st.session_state.my_ratings:
            list_df = pd.DataFrame([
                {"Title": v["title"], "Your ★": v["rating"]}
                for v in st.session_state.my_ratings.values()
            ])
            list_df.index = range(1, len(list_df) + 1)
            st.dataframe(list_df, use_container_width=True, height=220)

            # Remove a movie
            remove_choice = st.selectbox(
                "Remove a movie",
                options=["— select to remove —"] + [v["title"] for v in st.session_state.my_ratings.values()],
                key="foryou_remove_sel"
            )
            if remove_choice != "— select to remove —":
                if st.button("🗑 Remove", key="foryou_remove_btn"):
                    to_del = next(
                        (mid for mid, v in st.session_state.my_ratings.items()
                         if v["title"] == remove_choice), None
                    )
                    if to_del:
                        del st.session_state.my_ratings[to_del]
                        st.rerun()

            if st.button("🗑 Clear all", key="foryou_clear"):
                st.session_state.my_ratings = {}
                st.rerun()
        else:
            st.info("Add at least 3 movies to get personalised recommendations.")

    # ── Recommendations ──────────────────────────────────────────────────────
    if len(st.session_state.my_ratings) >= 1:
        st.divider()

        col_opt, _ = st.columns([1, 2])
        with col_opt:
            rec_model = st.selectbox(
                "Recommendation model",
                ["Content-Based"] + (["Neural CF (best accuracy)"] if ncf_available(models) else []),
                key="foryou_model"
            )

        if st.button("🎬 Get My Recommendations", type="primary", key="foryou_go"):
            rated_dict  = {mid: v["rating"] for mid, v in st.session_state.my_ratings.items()}
            exclude_ids = set(rated_dict.keys())

            with st.spinner('<span class="computing">Building your taste profile…</span>'):
                profile = build_profile(rated_dict, models)

            if profile is None:
                st.error("None of your movies are in the feature matrix. Try adding different titles.")
            else:
                with st.spinner("Finding your perfect matches…"):
                    recs = get_cb_recs_from_profile(profile, exclude_ids, models, n=n_recs)

                if recs.empty:
                    st.warning("No recommendations found.")
                else:
                    st.markdown(f'<div class="section-header fade-in">Your Top {len(recs)} Picks</div>', unsafe_allow_html=True)
                    st.dataframe(
                        _fmt(recs[["title","genres","year","cb_score","avg_rating","num_ratings"]])
                        .rename(columns=_RENAME),
                        use_container_width=True,
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · Letterboxd Import
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Import Your Letterboxd Ratings</div>', unsafe_allow_html=True)
    st.markdown("""
**How to export from Letterboxd:**
1. Go to letterboxd.com → your **Profile** → **Settings** → **Import & Export**
2. Click **Export Your Data** and download the zip
3. Extract and upload the **ratings.csv** file below

Your ratings stay local — nothing is sent anywhere.
    """)

    uploaded = st.file_uploader("Upload ratings.csv", type=["csv"])

    if uploaded is not None:
        try:
            lb_df = parse_letterboxd_csv(uploaded.read())
        except ValueError as exc:
            st.error(f"Could not parse the CSV: {exc}")
            st.stop()

        st.success(f"Found **{len(lb_df)} rated movies** in your export.")

        threshold = st.slider(
            "Match confidence threshold", min_value=0.60, max_value=0.99,
            value=0.80, step=0.05,
            help="Lower = more matches but more false positives.",
        )

        with st.spinner("Fuzzy-matching against 32,000 MovieLens titles…"):
            matched = match_to_movielens(lb_df, movies_df, threshold=threshold)

        n_matched = int(matched["matched"].sum())
        n_total   = len(matched)
        pct       = n_matched / n_total if n_total else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Films", n_total)
        c2.metric("Matched", n_matched)
        c3.metric("Match Rate", f"{pct:.0%}")

        col_m, col_u = st.columns(2)

        with col_m:
            st.markdown("**✅ Matched movies**")
            m_disp = (
                matched[matched["matched"]]
                [["title", "year", "rating", "ml_title", "match_score"]]
                .rename(columns=_RENAME | {"title": "Your Title"})
                .reset_index(drop=True)
            )
            m_disp.index = range(1, len(m_disp) + 1)
            st.dataframe(m_disp, use_container_width=True, height=280)

        with col_u:
            unmatched = matched[~matched["matched"]][["title", "year", "rating"]]
            if not unmatched.empty:
                st.markdown("**❌ Unmatched movies**")
                u_disp = unmatched.rename(columns=_RENAME | {"title": "Title"}).reset_index(drop=True)
                u_disp.index = range(1, len(u_disp) + 1)
                st.dataframe(u_disp, use_container_width=True, height=280)
            else:
                st.success("All movies were matched! 🎉")

        st.divider()

        if n_matched == 0:
            st.error("No movies matched — cannot generate recommendations.")
        else:
            if st.button("🎬 Generate My Recommendations", type="primary"):
                matched_ok  = matched[matched["matched"]].copy()
                rated_dict  = dict(zip(matched_ok["movieId"].astype(int), matched_ok["rating"]))
                exclude_ids = set(matched_ok["movieId"].astype(int))

                with st.spinner("Building taste profile and scoring movies…"):
                    profile = build_profile(rated_dict, models)

                if profile is None:
                    st.error("None of the matched movies are in the content feature matrix.")
                else:
                    recs = get_cb_recs_from_profile(profile, exclude_ids, models, n=n_recs)
                    st.markdown(f'<div class="section-header fade-in">Your Top {len(recs)} Picks</div>', unsafe_allow_html=True)
                    st.dataframe(_fmt(recs).rename(columns=_RENAME), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 · How It Works
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">How ReelRecs Works</div>', unsafe_allow_html=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    with st.expander("📊 The Dataset — MovieLens 25M", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown('<div class="metric-card"><div class="metric-value">25M</div><div class="metric-label">Ratings</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="metric-card"><div class="metric-value">162k</div><div class="metric-label">Users</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="metric-card"><div class="metric-value">32k</div><div class="metric-label">Movies</div></div>', unsafe_allow_html=True)
        c4.markdown('<div class="metric-card"><div class="metric-value">0.47%</div><div class="metric-label">Matrix Density</div></div>', unsafe_allow_html=True)
        st.markdown("""
The [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) contains 25 million ratings
applied to 62,000 movies by 162,000 users. After cleaning (minimum 5 ratings per movie, removing
outlier users), we work with **32,424 movies** and **162,112 users**.

Ratings range from 0.5 to 5.0 stars. The dataset was split 80/20 by timestamp — older ratings
train the models, newer ratings evaluate them.
        """)

    # ── CF ───────────────────────────────────────────────────────────────────
    with st.expander("🤝 Collaborative Filtering — SGD Matrix Factorisation"):
        st.markdown("""
**Idea:** Users who agreed in the past will agree in the future.

We factorise the 162k × 32k user-item rating matrix into low-dimensional embeddings:

```
R ≈ U · Vᵀ + user_bias + item_bias + global_mean
```

- **U** (162k × 100): a latent vector per user capturing their taste
- **V** (32k × 100): a latent vector per movie capturing its character
- Trained with **Stochastic Gradient Descent** minimising RMSE over 20 epochs

| Model | RMSE | MAE |
|---|---|---|
| Baseline (mean + bias) | 1.006 | 0.767 |
| SVD | 1.004 | 0.765 |
| **SGD-MF** | **1.002** | **0.768** |
| KNN Item-Based | 1.063 | 0.824 |

**Limitation:** Requires the user to be in the training data (cold-start problem).
        """)

    # ── CB ───────────────────────────────────────────────────────────────────
    with st.expander("📝 Content-Based Filtering — TF-IDF + Genres"):
        st.markdown("""
**Idea:** Recommend movies with similar content to ones you've liked.

Each movie is represented as a sparse feature vector combining:

- **TF-IDF** over user-generated tags (3,000 features, bigrams, sublinear TF)
- **Genre binary features** (20 genres, weighted ×2)
- **TMDB enrichment** (plot overview, director ×3, cast ×2, keywords) for top 5,000 movies

All vectors are L2-normalised so dot-product = cosine similarity.

**User profile:** a weighted average of feature vectors for movies the user has rated,
where the weight is `(rating − 3.0)` — so loved movies pull the profile towards them,
disliked movies push away.

**Strength:** Works for any user, even first-time visitors (no cold-start problem).
        """)

    # ── Neural CF ────────────────────────────────────────────────────────────
    with st.expander("🧠 Two-Tower Neural CF — Deep Learning"):
        st.markdown("""
**Idea:** Learn non-linear user and item representations with neural networks.

```
User ID ──► Embedding(64d) ──► Linear(128) ──► ReLU ──► Linear(32d) ──► ╗
                                                                          ╠──► dot ──► score
Item ID ──► Embedding(64d) ──► Linear(128) ──► ReLU ──► Linear(32d) ──► ╝
Genres  ──────────────────────────┘
```

- **12.7M parameters** trained on 20M ratings with Adam optimiser
- MSE loss computed on raw predictions (no clamping during training — clamping kills gradients)
- Global bias initialised to the dataset mean (3.52 ★) to prevent dead-gradient at start
- Early stopping via `ReduceLROnPlateau` on validation RMSE

| Epoch | Train RMSE | Val RMSE |
|---|---|---|
| 1 | 0.855 | 0.994 |
| 4 | 0.798 | **0.994** (best) |
| 10 | 0.765 | 0.998 |

**Best test RMSE: 0.9935** — the most accurate model in the project.
        """)

    # ── Hybrid ───────────────────────────────────────────────────────────────
    with st.expander("⚗️ Hybrid Model — Adaptive CF + CB Blend"):
        st.markdown("""
**Idea:** Use CF when we have enough data about a user; lean on CB otherwise.

```
score = α × CF_score + (1 − α) × CB_score
```

Alpha adapts to how much we know about the user:

| Ratings in history | α (CF weight) | Strategy |
|---|---|---|
| < 50 | 0.3 | Trust content similarity more |
| 50 – 200 | 0.6 | Balanced blend |
| > 200 | 0.8 | Trust collaborative signal more |

CB scores are calibrated to the rating scale via a linear fit from notebook 06:
```
CB_score = 0.2333 × cosine_similarity + 3.0967
```
        """)

    # ── Ranking models ───────────────────────────────────────────────────────
    with st.expander("📈 Ranking Models — BPR and ALS"):
        st.markdown("""
Besides rating prediction, we also trained **implicit feedback** models optimised
for ranking (rather than RMSE):

**BPR (Bayesian Personalised Ranking)** — pure NumPy, no compilation required
- Samples (user, positive item, negative item) triples
- Optimises: P(user prefers i over j) for all observed pairs
- Precision@10: **0.042** | nDCG@10: **0.042**

**ALS (Alternating Least Squares)** via the `implicit` library
- Confidence-weighted matrix factorisation: c_ui = 1 + 40 × rating
- Precision@10: 0.029 | nDCG@10: 0.030

These are better suited for "top-N list" tasks than the explicit rating models.
        """)

    # ── Tech stack ────────────────────────────────────────────────────────────
    with st.expander("🛠 Tech Stack"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Data & Models**
- Python · NumPy · SciPy · pandas
- scikit-learn (TF-IDF, SVD, normalisation)
- PyTorch (Two-Tower Neural CF)
- `implicit` (ALS)
- TMDB API (plot/cast enrichment)
            """)
        with c2:
            st.markdown("""
**App & Infrastructure**
- Streamlit (this dashboard)
- rapidfuzz (Letterboxd fuzzy matching)
- MovieLens 25M dataset
- Notebooks 01–09 (exploration → training)
- ~500 MB of trained model artefacts
            """)
