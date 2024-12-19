# src/models/hybrid.py
import numpy as np
import pandas as pd
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, content_weight=0.5):
        """
        Initialize hybrid recommender
        
        Parameters:
        content_weight (float): Weight for content-based recommendations (0-1)
        """
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.content_weight = content_weight
        self.collab_weight = 1 - content_weight
        
    def fit(self, movies_df, ratings_df):
        """
        Train both recommender systems
        
        Parameters:
        movies_df (pd.DataFrame): DataFrame containing movie information
        ratings_df (pd.DataFrame): DataFrame containing user ratings
        """
        # Train content-based system
        self.content_recommender.fit(movies_df)
        
        # Train collaborative system
        self.collaborative_recommender.fit(ratings_df, movies_df)
        
    def get_recommendations(self, user_id, movie_title, n_recommendations=5):
        """
        Get hybrid recommendations combining both approaches
        
        Parameters:
        user_id (int): User ID to get recommendations for
        movie_title (str): Movie title to base content recommendations on
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        pd.Series: Series of recommended movie titles
        """
        # Get recommendations from both systems
        content_recs, content_scores = self.content_recommender.get_recommendations(
            movie_title,
            n_recommendations=n_recommendations
        )
        
        collab_recs = self.collaborative_recommender.get_recommendations(
            user_id,
            n_recommendations=n_recommendations
        )
        
        # Create dictionaries of scores
        content_dict = dict(zip(content_recs, content_scores))
        
        # Get predicted ratings for collaborative recommendations
        collab_scores = []
        for title in collab_recs:
            movie_id = self.collaborative_recommender.movies_df[
                self.collaborative_recommender.movies_df['title'] == title
            ]['movieId'].iloc[0]
            score = self.collaborative_recommender.model.predict(
                user_id,
                movie_id
            ).est
            collab_scores.append(score)
        
        collab_dict = dict(zip(collab_recs, collab_scores))
        
        # Normalize scores
        def normalize_scores(scores):
            min_score = min(scores)
            max_score = max(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        content_scores = normalize_scores(content_scores)
        collab_scores = normalize_scores(collab_scores)
        
        # Combine recommendations
        all_movies = set(content_recs) | set(collab_recs)
        hybrid_scores = {}
        
        for movie in all_movies:
            # Calculate weighted score
            content_score = content_dict.get(movie, 0)
            collab_score = collab_dict.get(movie, 0)
            
            hybrid_scores[movie] = (
                self.content_weight * content_score +
                self.collab_weight * collab_score
            )
        
        # Sort by hybrid score
        sorted_recommendations = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N recommendations
        return pd.Series([r[0] for r in sorted_recommendations[:n_recommendations]])
    
    def explain_recommendations(self, user_id, movie_title, recommended_titles):
        """
        Provide explanations for hybrid recommendations
        
        Parameters:
        user_id (int): User ID
        movie_title (str): Original movie title
        recommended_titles (list): List of recommended movie titles
        
        Returns:
        dict: Dictionary containing explanations for each recommendation
        """
        explanations = {}
        
        for rec_title in recommended_titles:
            # Get content-based explanation
            content_explanation = self.content_recommender.explain_recommendation(
                movie_title,
                rec_title
            )
            
            # Get collaborative explanation
            movie_id = self.collaborative_recommender.movies_df[
                self.collaborative_recommender.movies_df['title'] == rec_title
            ]['movieId'].iloc[0]
            
            predicted_rating = self.collaborative_recommender.model.predict(
                user_id,
                movie_id
            ).est
            
            similar_users = self.collaborative_recommender.get_similar_users(
                user_
