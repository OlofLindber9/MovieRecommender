"""Letterboxd CSV import and fuzzy matching to MovieLens.

Letterboxd export steps:
  Profile → Settings → Import & Export → Export Your Data
  Unzip the archive and upload ratings.csv here.

Expected CSV columns: Date, Name, Year, Letterboxd URI, Rating
"""
from __future__ import annotations

import io
import re
import pandas as pd
from rapidfuzz import process, fuzz


def parse_letterboxd_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse a Letterboxd ratings export into a clean DataFrame.

    Returns columns: title (str), year (Int64), rating (float 0.5-5.0).
    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()

    # Some exports use "Rating10" (1-10 scale) instead of "Rating"
    if "Rating" not in df.columns and "Rating10" in df.columns:
        df["Rating"] = df["Rating10"] / 2.0

    required = {"Name", "Rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in Letterboxd CSV: {missing}. "
            f"Found: {list(df.columns)}"
        )

    result = df[["Name", "Year", "Rating"]].copy() if "Year" in df.columns \
        else df[["Name", "Rating"]].assign(Year=pd.NA)
    result.columns = ["title", "year", "rating"]

    result = result.dropna(subset=["rating"])
    result["year"]   = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    return result.reset_index(drop=True)


def _strip_year(title: str) -> str:
    """Remove trailing '(YYYY)' from a MovieLens title."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def match_to_movielens(
    lb_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Fuzzy-match Letterboxd titles to MovieLens movies.

    Adds columns to lb_df:
      movieId     – matched MovieLens ID (or NaN)
      ml_title    – matched MovieLens title string
      match_score – 0-1 confidence
      matched     – bool: True if score >= threshold
    """
    ml_titles  = movies_df["title"].tolist()
    ml_clean   = [_strip_year(t) for t in ml_titles]
    mid_list   = movies_df["movieId"].tolist()
    year_list  = (movies_df["year"].tolist()
                  if "year" in movies_df.columns
                  else [None] * len(ml_titles))

    results = []
    for _, row in lb_df.iterrows():
        query = str(row["title"]).strip()
        year  = row.get("year")

        hits = process.extract(
            query, ml_clean,
            scorer=fuzz.WRatio,
            limit=5, score_cutoff=50,
        )
        if not hits:
            results.append({"movieId": None, "ml_title": None,
                             "match_score": 0.0, "matched": False})
            continue

        # Prefer a candidate whose year also matches
        best_idx, best_score = hits[0][2], hits[0][1]
        if year and not pd.isna(year):
            for h_text, h_score, h_idx in hits:
                ml_yr = year_list[h_idx]
                if ml_yr is not None and not pd.isna(ml_yr):
                    try:
                        if abs(float(ml_yr) - float(year)) <= 1:
                            best_idx, best_score = h_idx, h_score
                            break
                    except (TypeError, ValueError):
                        pass

        score_norm = best_score / 100.0
        results.append({
            "movieId":     mid_list[best_idx] if score_norm >= threshold else None,
            "ml_title":    ml_titles[best_idx],
            "match_score": round(score_norm, 3),
            "matched":     score_norm >= threshold,
        })

    match_df = pd.DataFrame(results)
    return pd.concat([lb_df.reset_index(drop=True), match_df], axis=1)
