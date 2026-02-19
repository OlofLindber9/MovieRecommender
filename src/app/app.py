"""Movie Recommender — Streamlit Dashboard

Run from the project root:
    streamlit run src/app/app.py
"""
import sys
from pathlib import Path

# Ensure src/app/ is on sys.path so sibling modules resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lazy-import so the page config call happens first
from recommender import (
    load_all,
    search_movies,
    get_similar_movies,
    get_cf_recs,
    get_hybrid_recs,
    build_profile,
    get_cb_recs_from_profile,
)
from letterboxd import parse_letterboxd_csv, match_to_movielens


# ── Load models ───────────────────────────────────────────────────────────────
try:
    models = load_all()
    models_ok = True
except Exception as exc:
    st.error(
        f"**Could not load models:** {exc}\n\n"
        "Make sure you have run all six notebooks (01 – 06) first."
    )
    models_ok = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.markdown(
        """
Built on **MovieLens 25M** (25 million ratings, 62 k movies).

**Models available**
- SGD Matrix Factorisation (CF)
- TF-IDF + Genre Content-Based
- Adaptive Hybrid blend

**Tabs**
1. Find movies similar to one you love
2. Get personalised recs by user ID
3. Import your Letterboxd export
        """
    )
    st.divider()
    n_recs = st.slider("Recommendations to show", 5, 50, 20, step=5)

if not models_ok:
    st.stop()

movies_df  = models["movies_df"]
ratings_df = models["ratings_df"]


# ── Helper: format a results DataFrame for display ────────────────────────────
def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        out["year"] = out["year"].apply(
            lambda x: int(x) if pd.notna(x) else ""
        )
    for col in ("avg_rating",):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
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
    "avg_rating":       "Avg Rating",
    "num_ratings":      "# Ratings",
    "similarity":       "Similarity",
    "predicted_rating": "Predicted ★",
    "hybrid_score":     "Score",
    "cf_score":         "CF Score",
    "cb_score":         "CB Score",
    "ml_title":         "ML Title",
    "match_score":      "Match",
    "rating":           "Your ★",
}


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Movie Search", "👤 For You", "📋 Letterboxd"])


# ── Tab 1: Movie Search / Similar Movies ─────────────────────────────────────
with tab1:
    st.header("Find Similar Movies")
    st.caption(
        "Search for any movie and we'll find the most similar ones using "
        "TF-IDF tag embeddings + genre features."
    )

    query = st.text_input("Movie title", placeholder="e.g. The Dark Knight")

    if query:
        matches = search_movies(query, movies_df, n=10)

        if matches.empty:
            st.warning("No movies found — try a different spelling.")
        else:
            left, right = st.columns([1, 2])

            with left:
                st.markdown("**Matching titles:**")
                choice = st.radio(
                    "Select", matches["title"].tolist(), label_visibility="collapsed"
                )

            sel_row  = matches[matches["title"] == choice].iloc[0]
            movie_id = int(sel_row["movieId"])

            with right:
                st.markdown(f"### {choice}")
                c1, c2, c3 = st.columns(3)
                yr = sel_row.get("year")
                c1.metric("Year",       int(yr) if pd.notna(yr) else "N/A")
                c2.metric("Avg Rating", f"{sel_row.get('avg_rating', 0):.2f} ★")
                c3.metric("# Ratings",  f"{int(sel_row.get('num_ratings', 0)):,}")
                st.caption(f"Genres: {sel_row['genres']}")

            st.divider()
            st.markdown(f"#### Movies similar to *{choice}*")

            with st.spinner("Computing cosine similarity…"):
                similar = get_similar_movies(movie_id, models, n=n_recs)

            if similar.empty:
                st.info("This movie isn't in the content feature matrix.")
            else:
                st.dataframe(
                    _fmt(similar).rename(columns=_RENAME),
                    use_container_width=True,
                )


# ── Tab 2: For You (existing MovieLens users) ─────────────────────────────────
with tab2:
    st.header("Personalised Recommendations")
    st.caption(
        "Enter a MovieLens user ID to get recommendations tailored to your rating history. "
        "Valid IDs range from 1 to ~162,000."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        user_id = st.number_input(
            "User ID", min_value=1, max_value=500_000, value=1, step=1
        )
        model_choice = st.selectbox(
            "Model",
            ["Hybrid (recommended)", "Collaborative Filtering", "Content-Based"],
        )
        run_btn = st.button("Get Recommendations", type="primary")

    with col_b:
        user_hist = ratings_df[ratings_df["userId"] == user_id].head(15)
        if not user_hist.empty:
            hist_disp = (
                user_hist
                .merge(movies_df[["movieId", "title"]], on="movieId", how="left")
                [["title", "rating"]]
                .rename(columns={"title": "Movie", "rating": "★"})
            )
            hist_disp.index = range(1, len(hist_disp) + 1)
            st.markdown(f"**Sample ratings for user {user_id}:**")
            st.dataframe(hist_disp, use_container_width=True, height=230)
        else:
            st.info(f"User {user_id} not found. Try a different ID.")

    if run_btn:
        if user_id not in models["mappings"]["user_id_map"]:
            st.error(f"User {user_id} is not in the training data.")
        else:
            label = model_choice.split(" ")[0]
            with st.spinner(f"Generating {label} recommendations…"):
                if "Hybrid" in model_choice:
                    recs      = get_hybrid_recs(user_id, models, n=n_recs)
                    score_col = "hybrid_score"
                elif "Collaborative" in model_choice:
                    recs      = get_cf_recs(user_id, models, n=n_recs)
                    score_col = "predicted_rating"
                else:
                    user_rat   = ratings_df[ratings_df["userId"] == user_id]
                    rated_dict = dict(zip(user_rat["movieId"], user_rat["rating"]))
                    profile    = build_profile(rated_dict, models)
                    if profile is None:
                        st.error("Could not build a content profile for this user.")
                        st.stop()
                    recs      = get_cb_recs_from_profile(
                        profile, set(rated_dict), models, n=n_recs
                    )
                    score_col = "cb_score"

            if recs.empty:
                st.warning("No recommendations could be generated.")
            else:
                st.markdown(f"#### Top {len(recs)} recommendations")
                # Re-order: score column first
                cols_order = (
                    ["title", "genres", "year", score_col, "avg_rating", "num_ratings"]
                    + [c for c in recs.columns
                       if c not in ("title", "genres", "year", score_col,
                                    "avg_rating", "num_ratings")]
                )
                cols_order = [c for c in cols_order if c in recs.columns]
                st.dataframe(
                    _fmt(recs[cols_order]).rename(columns=_RENAME),
                    use_container_width=True,
                )


# ── Tab 3: Letterboxd Import ──────────────────────────────────────────────────
with tab3:
    st.header("Import Your Letterboxd Ratings")
    st.markdown(
        """
**How to export from Letterboxd:**
1. Go to letterboxd.com → your **Profile** → **Settings** → **Import & Export**
2. Click **Export Your Data** and download the zip
3. Extract and upload the **ratings.csv** file below

Your ratings are used only to build a local taste profile — nothing is sent anywhere.
        """
    )

    uploaded = st.file_uploader("Upload ratings.csv", type=["csv"])

    if uploaded is not None:
        # Parse
        try:
            lb_df = parse_letterboxd_csv(uploaded.read())
        except ValueError as exc:
            st.error(f"Could not parse the CSV: {exc}")
            st.stop()

        st.success(f"Found **{len(lb_df)} rated movies** in your export.")

        # Match
        threshold = st.slider(
            "Match confidence threshold",
            min_value=0.60, max_value=0.99,
            value=0.80, step=0.05,
            help="Lower = more matches but more false positives.",
        )

        with st.spinner("Fuzzy-matching against 62,000 MovieLens titles…"):
            matched = match_to_movielens(lb_df, movies_df, threshold=threshold)

        n_matched = int(matched["matched"].sum())
        n_total   = len(matched)
        pct       = n_matched / n_total if n_total else 0
        st.info(
            f"Matched **{n_matched} / {n_total}** movies ({pct:.0%}). "
            "Unmatched movies are excluded from recommendations."
        )

        col_m, col_u = st.columns(2)

        with col_m:
            st.markdown("**Matched movies**")
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
                st.markdown("**Unmatched movies**")
                u_disp = unmatched.rename(columns=_RENAME | {"title": "Title"}).reset_index(drop=True)
                u_disp.index = range(1, len(u_disp) + 1)
                st.dataframe(u_disp, use_container_width=True, height=280)
            else:
                st.success("All movies were matched!")

        st.divider()

        if n_matched == 0:
            st.error("No movies matched — cannot generate recommendations.")
        else:
            if st.button("Generate My Recommendations", type="primary"):
                matched_ok  = matched[matched["matched"]].copy()
                rated_dict  = dict(
                    zip(matched_ok["movieId"].astype(int), matched_ok["rating"])
                )
                exclude_ids = set(matched_ok["movieId"].astype(int))

                with st.spinner("Building taste profile and scoring movies…"):
                    profile = build_profile(rated_dict, models)

                if profile is None:
                    st.error(
                        "None of the matched movies are in the content feature matrix. "
                        "Try lowering the match threshold."
                    )
                else:
                    recs = get_cb_recs_from_profile(
                        profile, exclude_ids, models, n=n_recs
                    )
                    st.markdown(f"#### Your Top {len(recs)} Picks")
                    st.dataframe(
                        _fmt(recs).rename(columns=_RENAME),
                        use_container_width=True,
                    )
