# setup.py
from setuptools import setup, find_packages
setup(
    name="movie-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'nltk>=3.6.2',
        'surprise>=1.1.1',
        'requests>=2.26.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-mock>=3.6.1',
            'black>=21.7b0',
            'flake8>=3.9.2',
        ],
    },
    author="Harivdan N",
    author_email="harryshastri21@gmail.com",
    description="A hybrid movie recommendation system",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harryongit/movie_recommendation_system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
