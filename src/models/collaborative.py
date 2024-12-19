# src/models/collaborative.py
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from collections import defaultdict

class CollaborativeRecommender:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Initialize collaborative filtering recommender
        
        Parameters:
        n_factors (int): Number of latent factors
        n_epochs (int): Number of epochs for training
        lr_all (float): Learning rate for all parameters
        reg_all (float): Regularization term for all parameters
        """
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all
        )
        self.reader = Reader(rating_scale=(1, 5))
        self.trainset = None
        self.movies_df = None
        self.ratings_df = None
        
    def fit(self, ratings_df, movies_df):
        """
        Train the collaborative filtering model
        
        Parameters:
        ratings_df (pd.DataFrame): DataFrame containing user ratings
        movies_df (pd.DataFrame): DataFrame containing movie information
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
        # Create Surprise dataset
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']],
            self.reader
        )
        
        # Build full training set
        self.trainset = data.build_full_trainset()
        
        # Train the model
        self.model.fit(self.trainset)
        
    def get_recommendations(self, user_id, n_recommendations=5):
        """
        Get personalized recommendations for a user
        
        Parameters:
        user_id (int): User ID to get recommendations for
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        pd.Series: Series of recommended movie titles
        """
        # Get all movies
        all_movies = self.movies_df['movieId'].unique()
        
        # Get movies the user hasn't rated
        user_ratings = self.ratings_df[
            self.ratings_df['userId'] == user_id
        ]['movieId']
        movies_to_predict = np.setdiff1d(all_movies, user_ratings)
        
        # Make predictions
        predictions = []
        for movie_id in movies_to_predict:
            predicted_rating = self.model.predict(user_id, movie_id).est
            predictions.append((movie_id, predicted_rating))
        
        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N movie IDs
        top_movie_ids = [p[0] for p in predictions[:n_recommendations]]
        
        # Return movie titles
        return self.movies_df[
            self.movies_df['movieId'].isin(top_movie_ids)
        ]['title']
    
    def get_similar_users(self, user_id, n_similar=5):
        """
        Find similar users based on rating patterns
        
        Parameters:
        user_id (int): User ID to find similar users for
        n_similar (int): Number of similar users to return
        
        Returns:
        list: List of similar user IDs and their similarity scores
        """
        # Get user factors
        user_factors = self.model.pu[self.trainset.to_inner_uid(user_id)]
        
        # Calculate similarity with all users
        similarities = []
        for other_id in self.trainset.all_users():
            if other_id != user_id:
                other_factors = self.model.pu[other_id]
                similarity = np.dot(user_factors, other_factors) / (
                    np.linalg.norm(user_factors) * np.linalg.norm(other_factors)
                )
                similarities.append((self.trainset.to_raw_uid(other_id), similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def evaluate(self, cv_folds=5):
        """
        Evaluate the model using cross-validation
        
        Parameters:
        cv_folds (int): Number of folds for cross-validation
        
        Returns:
        dict: Dictionary containing evaluation metrics
        """
        # Create dataset
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']],
            self.reader
        )
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model,
            data,
            measures=['RMSE', 'MAE'],
            cv=cv_folds,
            verbose=False
        )
        
        return {
            'rmse_mean': cv_results['test_rmse'].mean(),
            'rmse_std': cv_results['test_rmse'].std(),
            'mae_mean': cv_results['test_mae'].mean(),
            'mae_std': cv_results['test_mae'].std()
        }
    
    def get_top_n_movies(self, n=10):
        """
        Get top N movies based on average ratings
        
        Parameters:
        n (int): Number of movies to return
        
        Returns:
        pd.DataFrame: DataFrame containing top rated movies
        """
        # Calculate average ratings and number of ratings
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        # Flatten column names
        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Filter movies with minimum number of ratings
        min_ratings = self.ratings_df['movieId'].value_counts().mean() * 0.5
        movie_stats = movie_stats[movie_stats['rating_count'] >= min_ratings]
        
        # Sort by average rating and get top N
        top_movies = movie_stats.nlargest(n, 'avg_rating')
        
        # Merge with movie information
        return pd.merge(
            top_movies,
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId'
        )
