# CineMatch 🎬

A state-of-the-art **Hybrid Movie Recommendation System** built using advanced machine learning techniques to provide highly personalized movie suggestions. This system combines content-based filtering and collaborative filtering approaches to ensure accurate and tailored movie recommendations. With the use of the **MovieLens dataset**, a leading benchmark in recommendation system research, this system offers a robust solution for both academic exploration and production-level deployment.

Whether you're looking for movies similar to your favorites, discovering movies based on user behavior, or benefiting from a hybrid approach combining multiple strategies, CineMatch covers all your needs.

![42195739-27dc-44c9-8a13-9e5fa16da84c](https://github.com/user-attachments/assets/060411ae-31c8-4aab-9ba0-ea2c0ae3a44a)


## Key Features


### 1. **Hybrid Approach**

* The system leverages both **content-based filtering** (analyzing movie features like genre, cast, director, etc.) and **collaborative filtering** (based on user-item interactions and behaviors) to provide personalized recommendations.

### 2. **Multiple Recommendation Strategies**

* **Content-Based**: Recommends movies that are similar to the ones you like based on movie features.
* **Collaborative Filtering**: Suggests movies based on the behavior and preferences of users with similar tastes.
* **Hybrid Approach**: Combines the strength of both content-based and collaborative filtering to enhance recommendation accuracy.

### 3. **Scalability and Flexibility**

* Built to handle datasets of varying sizes and adaptable to incorporate additional features such as **actor/actress** information, **director**, and **user feedback** for real-time recommendations.
* Easily customizable parameters for fine-tuning the system according to user requirements.

### 4. **Professional-Grade Implementation**

* Designed with **modular components** to separate concerns, **evaluation metrics** for assessing recommendation quality, and **comprehensive documentation** for ease of understanding and future development.
* Ready for deployment in **production environments** with efficient and scalable solutions.

### 5. **Future-Ready**

* Extensible for integration with **deep learning-based methods** and support for **API integration**, enabling real-time movie recommendations and future advancements in the recommendation system field.

---

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Quick Start

To get started quickly with CineMatch, follow these steps:

### 1. Clone the repository and set up the environment:

```bash
git clone https://github.com/harryongit/CineMatch.git
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate  
pip install -r requirements.txt  
```

### 2. Use the system:

```python
from src.models.recommender import MovieRecommender  

# Initialize the recommender system
recommender = MovieRecommender()

# Load dataset
recommender.load_data()

# Train models for content-based and collaborative filtering
recommender.train_content_based()
recommender.train_collaborative()

# Get hybrid recommendations
recommendations = recommender.get_hybrid_recommendations(
    user_id=42, movie_title="Inception", n_recommendations=5  
)

print(recommendations)  # Display the top 5 movie recommendations
```

---

## Project Structure

This project is organized into several key directories and files to ensure modularity and ease of use:

* **`data/`**
  Contains raw and processed datasets used in the recommendation system.

* **`src/`**
  Source code for model training, recommendation algorithms, and utilities. Includes:

  * `data_loader.py`: For loading and preprocessing the MovieLens data.
  * `features/`: Scripts for feature engineering, including data cleaning and transformations.
  * `models/`: Contains models for collaborative filtering, content-based filtering, and hybrid approaches.

    * `collaborative.py`: Implements collaborative filtering techniques.
    * `content_based.py`: Implements content-based filtering techniques.
    * `hybrid.py`: Combines content-based and collaborative filtering methods into a hybrid model.
    * `recommender.py`: Main class responsible for managing and orchestrating recommendation processes.
  * `utils/`: Utility scripts for evaluation, logging, and configuration.
  * `evaluation.py`: Contains functions for evaluating the performance of the recommendation system.
  * `app.py`: Main entry point for running the application in a production setting.

* **`tests/`**
  Unit tests to ensure the reliability and accuracy of the recommendation system.

* **`notebooks/`**
  Jupyter Notebooks for data exploration, visualization, and analysis.

* **`LICENSE`**
  The license file for the project, distributed under the MIT License.

* **`README.md`**
  This file, providing documentation and instructions for setting up and using the system.

* **`requirements.txt`**
  Python dependencies required for running the system.

* **`setup.py`**
  Setup script for packaging and installing the system as a Python module.

---

## Usage Examples

Here are a few examples of how to use CineMatch:

### **Content-Based Recommendations**

Find movies similar to a given movie based on content (such as genre, director, cast):

```python
recommender.get_content_based_recommendations("The Matrix", 5)
```

This will return the top 5 movies similar to **"The Matrix"** based on content features.

### **Collaborative Filtering Recommendations**

Get movie recommendations based on a user's past preferences and the preferences of similar users:

```python
recommender.get_collaborative_recommendations(42, 5)
```

This will return the top 5 movie recommendations for user **42** based on collaborative filtering.

### **Hybrid Recommendations**

Combine both content-based and collaborative filtering for more robust recommendations:

```python
recommender.get_hybrid_recommendations(42, "Inception", 5)
```

This will return the top 5 hybrid recommendations for user **42** based on the movie **"Inception"**.

---

## Contributing

We welcome contributions to improve and extend CineMatch. To contribute:

1. Fork the repository and create a feature branch:

```bash
git checkout -b feature/AmazingFeature
```

2. Make your changes and push them to your fork.

3. Open a Pull Request describing your changes.

We follow a code review process to ensure that contributions meet project standards.

---

## License

This project is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For any queries or issues, feel free to contact me:

* **Harry**: [GitHub Profile](https://github.com/harryongit)

---

⭐️ **Star the repository** if you find it useful!
