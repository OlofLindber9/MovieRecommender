"""
Download and extract MovieLens 25M dataset.

Usage:
    python scripts/download_data.py
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_file(url, destination):
    """Download a file with progress indication."""
    print(f"Downloading from {url}...")
    print(f"Saving to {destination}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='')

    print("\nDownload complete!")


def extract_zip(zip_path, extract_to):
    """Extract ZIP file to destination directory."""
    print(f"Extracting {zip_path} to {extract_to}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print("Extraction complete!")


def verify_dataset(data_dir):
    """Verify that all required files exist."""
    required_files = ['ratings.csv', 'movies.csv', 'links.csv', 'tags.csv', 'README.txt']

    print("\nVerifying dataset files...")
    all_exist = True

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_exist = False

    return all_exist


def main():
    """Main function to download and extract MovieLens dataset."""
    print("MovieLens 25M Dataset Downloader")
    print("=" * 50)

    # Load configuration
    config = load_config()
    url = config['data']['movielens']['url']
    raw_dir = project_root / config['data']['raw_dir']

    # Create raw directory if it doesn't exist
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    zip_path = raw_dir / "ml-25m.zip"
    extract_dir = raw_dir
    dataset_dir = raw_dir / "ml-25m"

    # Check if dataset already exists
    if dataset_dir.exists() and verify_dataset(dataset_dir):
        print(f"\nDataset already exists at {dataset_dir}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return

    # Download dataset
    try:
        download_file(url, zip_path)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        sys.exit(1)

    # Extract dataset
    try:
        extract_zip(zip_path, extract_dir)
    except Exception as e:
        print(f"\nError extracting dataset: {e}")
        sys.exit(1)

    # Verify extraction
    if verify_dataset(dataset_dir):
        print("\n✓ Dataset successfully downloaded and verified!")
        print(f"\nDataset location: {dataset_dir}")
        print("\nNext steps:")
        print("  1. Open notebooks/01_data_exploration.ipynb to explore the data")
        print("  2. Run notebooks/02_data_cleaning.ipynb to clean the data")
    else:
        print("\n✗ Dataset verification failed!")
        sys.exit(1)

    # Clean up ZIP file
    if zip_path.exists():
        response = input("\nDelete ZIP file to save space? (Y/n): ")
        if response.lower() != 'n':
            zip_path.unlink()
            print("ZIP file deleted.")


if __name__ == "__main__":
    main()
