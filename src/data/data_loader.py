# src/data/data_loader.py
import pandas as pd
import os
import requests
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_movielens_data(self):
        """Download MovieLens dataset if not present"""
        base_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        
        if not (self.raw_dir / "movies.csv").exists():
            print("Downloading MovieLens dataset...")
            response = requests.get(base_url)
            zip_path = self.raw_dir / "ml-latest-small.zip"
            
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            # Extract files
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
            
            # Remove zip file
            zip_path.unlink()
            
        return self.load_data()
    
    def load_data(self):
        """Load movies and ratings data"""
        movies_path = self.raw_dir / "movies.csv"
        ratings_path = self.raw_dir / "ratings.csv"
        
        if not movies_path.exists() or not ratings_path.exists():
            raise FileNotFoundError("Dataset files not found. Run download_movielens_data() first.")
        
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        
        return movies_df, ratings_df
    
    def preprocess_data(self, movies_df, ratings_df):
        """Preprocess the data for recommendation"""
        # Clean movie titles
        movies_df['title'] = movies_df['title'].str.strip()
        
        # Convert genres to list
        movies_df['genres'] = movies_df['genres'].str.split('|')
        
        # Create user-movie rating matrix
        rating_matrix = ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Save processed data
        movies_df.to_csv(self.processed_dir / "processed_movies.csv", index=False)
        rating_matrix.to_csv(self.processed_dir / "rating_matrix.csv")
        
        return movies_df, rating_matrix
