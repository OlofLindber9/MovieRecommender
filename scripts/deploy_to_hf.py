#!/usr/bin/env python3
"""Deploy ReelRecs to Hugging Face Spaces.

Prerequisites
-------------
1. Install the HF Hub client::

    pip install huggingface_hub

2. Log in once (you'll be prompted for your HF token)::

    huggingface-cli login

   Get a token at https://huggingface.co/settings/tokens  (write access)

3. Run this script::

    python scripts/deploy_to_hf.py --repo-id YOUR_USERNAME/reelrecs

The Space will be created if it doesn't exist.  All model and data files are
uploaded via Git LFS (handled automatically by huggingface_hub).  First build
can take 3–5 minutes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../movie-recommender/

# ── Files to upload ────────────────────────────────────────────────────────
CODE_FILES = [
    (ROOT / "src/app/app.py",         "src/app/app.py"),
    (ROOT / "src/app/recommender.py", "src/app/recommender.py"),
    (ROOT / "src/app/letterboxd.py",  "src/app/letterboxd.py"),
    (ROOT / "src/app/__init__.py",    "src/app/__init__.py"),
]

MODEL_FILES = [
    # Required for CF / hybrid recommendations
    (ROOT / "src/models/sgd_mf_model.pkl",     "src/models/sgd_mf_model.pkl"),
    # Content-based model (enriched preferred, plain as fallback)
    (ROOT / "src/models/cb_enriched_model.pkl", "src/models/cb_enriched_model.pkl"),
    (ROOT / "src/models/cb_model.pkl",          "src/models/cb_model.pkl"),
    # Neural CF (optional — skip-ncf flag omits these)
    (ROOT / "src/models/neural_cf_best.pt",     "src/models/neural_cf_best.pt"),
    (ROOT / "src/models/neural_cf_config.yaml", "src/models/neural_cf_config.yaml"),
]

DATA_FILES = [
    (ROOT / "src/data/features/id_mappings.pkl",
     "src/data/features/id_mappings.pkl"),
    (ROOT / "src/data/features/movie_features.parquet",
     "src/data/features/movie_features.parquet"),
    # ratings_cleaned.parquet is large (314 MB) — uploaded via LFS
    (ROOT / "src/data/processed/ratings_cleaned.parquet",
     "src/data/processed/ratings_cleaned.parquet"),
]

# ── Content for files we generate on-the-fly ──────────────────────────────
HF_DOCKERFILE = """\
FROM python:3.11-slim

WORKDIR /app

# System deps needed by scipy / numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU torch from PyTorch index, then the rest from PyPI
RUN pip install --no-cache-dir \\
    --extra-index-url https://download.pytorch.org/whl/cpu \\
    -r requirements.txt

COPY . .

EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["python", "-m", "streamlit", "run", "src/app/app.py", \\
     "--server.port=7860", "--server.address=0.0.0.0"]
"""

HF_README = """\
---
title: ReelRecs Movie Recommender
emoji: 🎬
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# ReelRecs 🎬

An AI-powered movie recommender built on the **MovieLens 25M** dataset
(25 M ratings · 162 k users · 32 k movies).

## Features

- **Movie Search** — find similar films using TF-IDF tag embeddings + genre features
- **For You** — personalised picks by rating a few movies you've seen
- **Letterboxd Import** — upload your Letterboxd export CSV for instant recs
- **How It Works** — interactive explainer for every model

## Models

| Model | RMSE | Notes |
|---|---|---|
| SGD Matrix Factorisation | 1.002 | Collaborative filtering baseline |
| Two-Tower Neural CF | **0.994** | Best accuracy |
| Content-Based (TF-IDF + TMDB) | — | No cold-start |
| Adaptive Hybrid CF + CB | — | Blends CF + CB by history size |
| BPR Ranking | P@10 = 0.042 | Ranking model |
| ALS Ranking | P@10 = 0.029 | Ranking model |

## Tech Stack

Python · NumPy · SciPy · pandas · scikit-learn · PyTorch · Streamlit ·
rapidfuzz · implicit · TMDB API
"""

HF_REQUIREMENTS = """\
# Core dependencies
streamlit>=1.38.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
rapidfuzz>=3.0.0
pyyaml>=6.0
pyarrow>=12.0.0

# PyTorch CPU build (avoids downloading the 2 GB CUDA wheel)
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
"""


# ── Helpers ────────────────────────────────────────────────────────────────
def _upload_text(api, content: str, path_in_repo: str, repo_id: str, desc: str):
    api.upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="space",
        commit_message=desc,
    )
    print(f"  ✓  {path_in_repo}")


def _upload_file(api, local: Path, path_in_repo: str, repo_id: str):
    if not local.exists():
        print(f"  ⚠  SKIP (not found locally): {local.name}")
        return
    mb = local.stat().st_size / 1_000_000
    print(f"  → uploading {path_in_repo}  ({mb:.1f} MB) …", flush=True)
    api.upload_file(
        path_or_fileobj=local,          # Path object — huggingface_hub handles LFS
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"upload {path_in_repo}",
    )
    print(f"  ✓  {path_in_repo}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deploy ReelRecs to HF Spaces")
    parser.add_argument(
        "--repo-id", required=True,
        help="HF Space repo id in the form  username/space-name  "
             "(e.g. alice/reelrecs)",
    )
    parser.add_argument(
        "--skip-ncf", action="store_true",
        help="Skip uploading Neural CF model files (saves ~49 MB)",
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip uploading ratings_cleaned.parquet (saves ~314 MB). "
             "The app will still work but 'For You' CF won't show user history.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the Space as private (default: public)",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face write token (from huggingface.co/settings/tokens). "
             "If omitted, uses the cached token from a prior login.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed.  Run:  pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    # ── Verify authentication ──────────────────────────────────────────────
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception:
        print(
            "ERROR: Not authenticated.\n"
            "Run:  python -c \"from huggingface_hub import login; login()\"\n"
            "  or: python scripts/deploy_to_hf.py --token YOUR_HF_TOKEN ..."
        )
        sys.exit(1)

    # ── Create Space if it doesn't exist ──────────────────────────────────
    print(f"\nCreating / verifying Space: {args.repo_id} …")
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print(f"  Space URL: https://huggingface.co/spaces/{args.repo_id}")
    except Exception as exc:
        print(f"ERROR creating Space: {exc}")
        sys.exit(1)

    # ── README + Dockerfile + requirements ───────────────────────────────
    print("\nUploading config …")
    _upload_text(api, HF_README,        "README.md",        args.repo_id, "Add HF Spaces README")
    _upload_text(api, HF_DOCKERFILE,    "Dockerfile",       args.repo_id, "Add Dockerfile")
    _upload_text(api, HF_REQUIREMENTS,  "requirements.txt", args.repo_id, "Add requirements.txt")

    # ── App code ──────────────────────────────────────────────────────────
    print("\nUploading app code …")
    for local, repo_path in CODE_FILES:
        _upload_file(api, local, repo_path, args.repo_id)

    # ── Models ────────────────────────────────────────────────────────────
    print("\nUploading models …")
    for local, repo_path in MODEL_FILES:
        if args.skip_ncf and "neural_cf" in repo_path:
            print(f"  ⚠  SKIP (--skip-ncf): {repo_path}")
            continue
        _upload_file(api, local, repo_path, args.repo_id)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nUploading data files …")
    for local, repo_path in DATA_FILES:
        if args.skip_data and "ratings_cleaned" in repo_path:
            print(f"  ⚠  SKIP (--skip-data): {repo_path}")
            continue
        _upload_file(api, local, repo_path, args.repo_id)

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════╗
║  Upload complete!                                    ║
║                                                      ║
║  Your Space will build in ~3 minutes:                ║
║  https://huggingface.co/spaces/{args.repo_id:<22}║
╚══════════════════════════════════════════════════════╝
""")
    print("Tip: watch the build logs in the Space's 'Logs' tab.")
    if args.skip_ncf:
        print("NOTE: Neural CF was not uploaded — the model selector won't show it.")
    if args.skip_data:
        print("NOTE: ratings_cleaned.parquet was not uploaded.")
        print("      The 'For You' tab still works via the content-based profile.")


if __name__ == "__main__":
    main()
