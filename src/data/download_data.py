# src/data/download_data.py
import requests
import zipfile
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_movielens_data(data_dir="data/raw"):
    """
    Download the MovieLens dataset
    
    Parameters:
    data_dir (str): Directory to save the dataset
    """
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # MovieLens dataset URL
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    try:
        # Download the dataset
        logger.info("Downloading MovieLens dataset...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save zip file
        zip_path = data_path / "ml-latest-small.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extract files
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        
        # Remove zip file
        zip_path.unlink()
        
        logger.info("Dataset downloaded and extracted successfully!")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        raise
    except zipfile.BadZipFile:
        logger.error("Error extracting zip file")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    download_movielens_data()
