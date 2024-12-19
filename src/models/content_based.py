# src/models/content_based.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self.movies_df = None
        
    def fit(self, movies_df):
        """
        Train the content-based model using movie features
        
        Parameters:
        movies_df (pd.DataFrame): DataFrame containing movie information
        """
        self.movies_df = movies_df
        
        # Create TF-IDF matrix from combined features
        tfidf_matrix = self.tfidf.fit_transform(movies_df['combined_features'])
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
    def get_recommendations(self, movie_title, n_recommendations=5):
        """
        Get movie recommendations based on content similarity
        
        Parameters:
        movie_title (str): Title of the movie to base recommendations on
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        pd.Series: Series of recommended movie titles
        """
        # Find movie index
        try:
            idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        except IndexError:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset")
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding the input movie)
        sim_scores = sim_scores[1:n_recommendations + 1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movie titles and their similarity scores
        recommendations = self.movies_df['title'].iloc[movie_indices]
        scores = [score for _, score in sim_scores]
        
        return recommendations, scores
    
    def get_similar_movies_by_genre(self, genres, n_recommendations=5):
        """
        Get movie recommendations based on specific genres
        
        Parameters:
        genres (list): List of genres to base recommendations on
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        pd.Series: Series of recommended movie titles
        """
        # Create a feature vector for the input genres
        genre_feature = ' '.join(genres)
        genre_vector = self.tfidf.transform([genre_feature])
        
        # Calculate similarity with all movies
        sim_scores = cosine_similarity(
            genre_vector,
            self.tfidf.transform(self.movies_df['combined_features'])
        ).flatten()
        
        # Get top N movies
        movie_indices = sim_scores.argsort()[-n_recommendations:][::-1]
        
        return self.movies_df['title'].iloc[movie_indices]
    
    def explain_recommendation(self, movie_title, recommended_title):
        """
        Explain why a movie was recommended
        
        Parameters:
        movie_title (str): Original movie title
        recommended_title (str): Recommended movie title
        
        Returns:
        dict: Dictionary containing explanation details
        """
        # Get indices for both movies
        idx1 = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        idx2 = self.movies_df[self.movies_df['title'] == recommended_title].index[0]
        
        # Get similarity score
        similarity = self.similarity_matrix[idx1][idx2]
        
        # Get common genres
        genres1 = set(self.movies_df.loc[idx1, 'genres'])
        genres2 = set(self.movies_df.loc[idx2, 'genres'])
        common_genres = genres1.intersection(genres2)
        
        return {
            'similarity_score': similarity,
            'common_genres': list(common_genres),
            'original_genres': list(genres1),
            'recommended_genres': list(genres2)
        }
