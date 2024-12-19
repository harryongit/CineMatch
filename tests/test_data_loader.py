# tests/test_data_loader.py
import pytest
import pandas as pd
from src.data.data_loader import DataLoader
from pathlib import Path

@pytest.fixture
def data_loader():
    return DataLoader(data_dir="test_data")

def test_create_directories(data_loader):
    """Test if directories are created correctly"""
    assert Path(data_loader.raw_dir).exists()
    assert Path(data_loader.processed_dir).exists()

def test_load_data(data_loader, mocker):
    """Test data loading functionality"""
    # Mock CSV files
    mock_movies = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'genres': ['Action', 'Comedy', 'Drama']
    })
    mock_ratings = pd.DataFrame({
        'userId': [1, 1, 2],
        'movieId': [1, 2, 3],
        'rating': [4.0, 3.5, 5.0]
    })
    
    # Mock read_csv
    mocker.patch('pandas.read_csv', side_effect=[mock_movies, mock_ratings])
    
    movies_df, ratings_df = data_loader.load_data()
    
    assert isinstance(movies_df, pd.DataFrame)
    assert isinstance(ratings_df, pd.DataFrame)
    assert len(movies_df) == 3
    assert len(ratings_df) == 3

# tests/test_content_based.py
import pytest
import pandas as pd
import numpy as np
from src.models.recommender import MovieRecommender

@pytest.fixture
def recommender():
    return MovieRecommender()

def test_content_based_recommendations(recommender, mocker):
    """Test content-based recommendation functionality"""
    # Mock data
    mock_movies = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'genres': ['Action', 'Comedy', 'Drama']
    })
    
    # Mock similarity matrix
    mock_sim = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    recommender.movies_df = mock_movies
    recommender.cosine_sim = mock_sim
    
    recs = recommender.get_content_based_recommendations('Movie 1', n_recommendations=2)
    
    assert len(recs) == 2
    assert 'Movie 1' not in recs.values

# tests/test_collaborative.py
import pytest
import pandas as pd
from src.models.recommender import MovieRecommender

@pytest.fixture
def recommender():
    return MovieRecommender()

def test_collaborative_recommendations(recommender, mocker):
    """Test collaborative filtering recommendation functionality"""
    # Mock data
    mock_movies = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'genres': ['Action', 'Comedy', 'Drama']
    })
    
    mock_ratings = pd.DataFrame({
        'userId': [1, 1, 2],
        'movieId': [1, 2, 3],
        'rating': [4.0, 3.5, 5.0]
    })
    
    recommender.movies_df = mock_movies
    recommender.ratings_df = mock_ratings
    
    # Mock SVD predictions
    class MockSVD:
        def predict(self, user_id, movie_id):
            class Prediction:
                def __init__(self, est):
                    self.est = est
            return Prediction(4.0)
    
    recommender.svd_model = MockSVD()
    
    recs = recommender.get_collaborative_recommendations(user_id=1, n_recommendations=2)
    
    assert len(recs) == 2
