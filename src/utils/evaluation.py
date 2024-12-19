# src/utils/evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import accuracy

class RecommenderEvaluation:
    @staticmethod
    def calculate_metrics(predictions, actual):
        """Calculate basic recommendation metrics"""
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae
        }
    
    @staticmethod
    def evaluate_surprise_predictions(predictions):
        """Evaluate predictions from Surprise library"""
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        return {
            'rmse': rmse,
            'mae': mae
        }
    
    @staticmethod
    def calculate_coverage(recommended_items, total_items):
        """Calculate catalog coverage"""
        return len(recommended_items) / len(total_items)
    
    @staticmethod
    def calculate_diversity(recommendations, similarity_matrix):
        """Calculate diversity of recommendations"""
        n_items = len(recommendations)
        if n_items <= 1:
            return 0.0
            
        sum_similarity = 0
        count = 0
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                sum_similarity += similarity_matrix[recommendations[i]][recommendations[j]]
                count += 1
                
        return 1 - (sum_similarity / count if count > 0 else 0)
    
    @staticmethod
    def calculate_novelty(recommendations, popularity_scores):
        """Calculate novelty of recommendations"""
        if not recommendations:
            return 0.0
            
        return -np.mean([np.log2(popularity_scores[item]) for item in recommendations])
    
    def evaluate_recommendations(self, recommender, test_users, n_recommendations=5):
        """Comprehensive evaluation of recommendations"""
        results = {
            'coverage': [],
            'diversity': [],
            'novelty': []
        }
        
        total_items = recommender.movies_df['movieId'].unique()
        popularity_scores = recommender.ratings_df['movieId'].value_counts(normalize=True)
        
        for user_id in test_users:
            # Get recommendations
            recs = recommender.get_collaborative_recommendations(
                user_id,
                n_recommendations=n_recommendations
            )
            rec_ids = recommender.movies_df[
                recommender.movies_df['title'].isin(recs)
            ]['movieId'].values
            
            # Calculate metrics
            results['coverage'].append(
                self.calculate_coverage(rec_ids, total_items)
            )
            results['diversity'].append(
                self.calculate_diversity(rec_ids, recommender.cosine_sim)
            )
            results['novelty'].append(
                self.calculate_novelty(rec_ids, popularity_scores)
            )
        
        # Average results
        return {metric: np.mean(values) for metric, values in results.items()}
