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
                user_id,
                n_similar=3
            )
            
            explanations[rec_title] = {
                'content_based': {
                    'similarity_score': content_explanation['similarity_score'],
                    'common_genres': content_explanation['common_genres']
                },
                'collaborative': {
                    'predicted_rating': predicted_rating,
                    'similar_users': similar_users
                },
                'hybrid_score': (
                    self.content_weight * content_explanation['similarity_score'] +
                    self.collab_weight * (predicted_rating / 5.0)
                )
            }
        
        return explanations
    
    def evaluate(self, test_users, test_movies):
        """
        Evaluate hybrid recommender performance
        
        Parameters:
        test_users (list): List of user IDs for testing
        test_movies (list): List of movie titles for testing
        
        Returns:
        dict: Dictionary containing evaluation metrics
        """
        content_metrics = []
        collab_metrics = []
        hybrid_metrics = []
        
        for user_id in test_users:
            for movie_title in test_movies:
                # Get recommendations from each system
                content_recs, _ = self.content_recommender.get_recommendations(movie_title)
                collab_recs = self.collaborative_recommender.get_recommendations(user_id)
                hybrid_recs = self.get_recommendations(user_id, movie_title)
                
                # Calculate precision and recall
                content_metrics.append(self._calculate_metrics(content_recs, test_movies))
                collab_metrics.append(self._calculate_metrics(collab_recs, test_movies))
                hybrid_metrics.append(self._calculate_metrics(hybrid_recs, test_movies))
        
        return {
            'content_based': {
                'precision': np.mean([m['precision'] for m in content_metrics]),
                'recall': np.mean([m['recall'] for m in content_metrics])
            },
            'collaborative': {
                'precision': np.mean([m['precision'] for m in collab_metrics]),
                'recall': np.mean([m['recall'] for m in collab_metrics])
            },
            'hybrid': {
                'precision': np.mean([m['precision'] for m in hybrid_metrics]),
                'recall': np.mean([m['recall'] for m in hybrid_metrics])
            }
        }
    
    def _calculate_metrics(self, recommended_items, relevant_items):
        """
        Calculate precision and recall metrics
        
        Parameters:
        recommended_items (list): List of recommended items
        relevant_items (list): List of relevant items
        
        Returns:
        dict: Dictionary containing precision and recall scores
        """
        recommended_set = set(recommended_items)
        relevant_set = set(relevant_items)
        
        true_positives = len(recommended_set.intersection(relevant_set))
        
        precision = true_positives / len(recommended_set) if recommended_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        
        return {'precision': precision, 'recall': recall}
