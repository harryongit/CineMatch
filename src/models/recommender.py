# src/models/recommender.py
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineering
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import pandas as pd
import numpy as np

class MovieRecommender:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineering = FeatureEngineering()
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.svd_model = None
        
    def load_data(self):
        """Load and preprocess the data"""
        # Download data if needed
        self.movies_df, self.ratings_df = self.data_loader.download_movielens_data()
        
        # Create features
        self.movies_df = self.feature_engineering.create_movie_features(self.movies_df)
        self.tfidf_matrix = self.feature_engineering.create_tfidf_matrix(self.movies_df)
        
    def train_content_based(self):
        """Train content-based filtering model"""
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def train_collaborative(self):
        """Train collaborative filtering model"""
        # Create Surprise reader and data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Train SVD model
        self.svd_model = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02
        )
        self.svd_model.fit(data.build_full_trainset())
        
    def get_content_based_recommendations(self, movie_title, n_recommendations=5):
        """Get content-based recommendations"""
        # Find movie index
        idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        return self.movies_df['title'].iloc[movie_indices]
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get collaborative filtering recommendations"""
        # Get all movies
        all_movies = self.movies_df['movieId'].unique()
        
        # Get movies user hasn't rated
        user_ratings = self.ratings_df[
            self.ratings_df['userId'] == user_id
        ]['movieId']
        movies_to_predict = np.setdiff1d(all_movies, user_ratings)
        
        # Make predictions
        predictions = [
            (movie_id, self.svd_model.predict(user_id, movie_id).est)
            for movie_id in movies_to_predict
        ]
        
        # Sort predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N movie IDs
        top_movie_ids = [p[0] for p in predictions[:n_recommendations]]
        
        return self.movies_df[
            self.movies_df['movieId'].isin(top_movie_ids)
        ]['title']
    
    def get_hybrid_recommendations(self, user_id, movie_title, n_recommendations=5):
        """Get hybrid recommendations"""
        content_recs = self.get_content_based_recommendations(
            movie_title,
            n_recommendations=n_recommendations
        )
        collab_recs = self.get_collaborative_recommendations(
            user_id,
            n_recommendations=n_recommendations
        )
        
        # Combine recommendations
        hybrid_recs = pd.concat([content_recs, collab_recs]).drop_duplicates()
        
        return hybrid_recs.head(n_recommendations)
