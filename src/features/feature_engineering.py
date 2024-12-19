# src/features/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk

class FeatureEngineering:
    def __init__(self):
        self.ps = PorterStemmer()
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def create_movie_features(self, movies_df):
        """Create features for content-based filtering"""
        # Combine relevant features
        movies_df['combined_features'] = movies_df.apply(
            lambda x: ' '.join([
                x['title'],
                ' '.join(x['genres']),
            ]), axis=1
        )
        
        # Apply text preprocessing
        movies_df['combined_features'] = movies_df['combined_features'].apply(
            self._preprocess_text
        )
        
        return movies_df
    
    def create_user_features(self, ratings_df):
        """Create user features for collaborative filtering"""
        # Calculate user statistics
        user_features = pd.DataFrame()
        
        # Average rating per user
        user_features['avg_rating'] = ratings_df.groupby('userId')['rating'].mean()
        
        # Number of ratings per user
        user_features['num_ratings'] = ratings_df.groupby('userId')['rating'].count()
        
        # Rating variance per user
        user_features['rating_variance'] = ratings_df.groupby('userId')['rating'].var()
        
        return user_features
    
    def _preprocess_text(self, text):
        """Preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenization
        tokens = nltk.word_tokenize(text)
        
        # Stemming
        tokens = [self.ps.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def create_tfidf_matrix(self, movies_df):
        """Create TF-IDF matrix from movie features"""
        tfidf_matrix = self.tfidf.fit_transform(movies_df['combined_features'])
        return tfidf_matrix
