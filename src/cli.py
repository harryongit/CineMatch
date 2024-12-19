# src/cli.py
import click
import pandas as pd
from models.recommender import MovieRecommender
from pathlib import Path
import json

@click.group()
def cli():
    """Movie Recommender System CLI"""
    pass

@cli.command()
@click.option('--movie', prompt='Movie title', help='Movie title to base recommendations on')
@click.option('--num', default=5, help='Number of recommendations')
def content_based(movie, num):
    """Get content-based movie recommendations"""
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.train_content_based()
    
    try:
        recommendations = recommender.get_content_based_recommendations(movie, num)
        click.echo("\nRecommended movies:")
        for i, title in enumerate(recommendations, 1):
            click.echo(f"{i}. {title}")
    except ValueError as e:
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--user', prompt='User ID', type=int, help='User ID to get recommendations for')
@click.option('--num', default=5, help='Number of recommendations')
def collaborative(user, num):
    """Get collaborative filtering recommendations"""
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.train_collaborative()
    
    try:
        recommendations = recommender.get_collaborative_recommendations(user, num)
        click.echo("\nRecommended movies:")
        for i, title in enumerate(recommendations, 1):
            click.echo(f"{i}. {title}")
    except ValueError as e:
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--user', prompt='User ID', type=int, help='User ID to get recommendations for')
@click.option('--movie', prompt='Movie title', help='Movie title to base recommendations on')
@click.option('--num', default=5, help='Number of recommendations')
def hybrid(user, movie, num):
    """Get hybrid recommendations"""
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.train_content_based()
    recommender.train_collaborative()
    
    try:
        recommendations = recommender.get_hybrid_recommendations(user, movie, num)
        click.echo("\nRecommended movies:")
        for i, title in enumerate(recommendations, 1):
            click.echo(f"{i}. {title}")
    except ValueError as e:
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--output', default='evaluation_results.json', help='Output file for evaluation results')
def evaluate(output):
    """Evaluate recommender system performance"""
    recommender = MovieRecommender()
    recommender.load_data()
    recommender.train_content_based()
    predictions = recommender.train_collaborative()
    
    # Get evaluation metrics
    from src.utils.evaluation import RecommenderEvaluation
    evaluator = RecommenderEvaluation()
    metrics = evaluator.evaluate_surprise_predictions(predictions)
    
    # Save results
    with open(output, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    click.echo(f"\nEvaluation results saved to {output}")
    click.echo(f"RMSE: {metrics['rmse']:.4f}")
    click.echo(f"MAE: {metrics['mae']:.4f}")

if __name__ == '__main__':
    cli()
