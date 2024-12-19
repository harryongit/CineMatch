# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from models.recommender import MovieRecommender
import uvicorn

app = FastAPI(
    title="Movie Recommender API",
    description="API for movie recommendations using hybrid recommender system",
    version="1.0.0"
)

# Initialize recommender
recommender = MovieRecommender()
recommender.load_data()
recommender.train_content_based()
recommender.train_collaborative()

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    movie_title: Optional[str] = None
    n_recommendations: int = 5

class RecommendationResponse(BaseModel):
    recommendations: List[str]
    explanation: Optional[dict] = None

@app.post("/recommend/content-based", response_model=RecommendationResponse)
async def content_based_recommendations(request: RecommendationRequest):
    """Get content-based recommendations"""
    try:
        recommendations = recommender.get_content_based_recommendations(
            request.movie_title,
            request.n_recommendations
        )
        return {"recommendations": recommendations.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend/collaborative", response_model=RecommendationResponse)
async def collaborative_recommendations(request: RecommendationRequest):
    """Get collaborative filtering recommendations"""
    try:
        recommendations = recommender.get_collaborative_recommendations(
            request.user_id,
            request.n_recommendations
        )
        return {"recommendations": recommendations.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend/hybrid", response_model=RecommendationResponse)
async def hybrid_recommendations(request: RecommendationRequest):
    """Get hybrid recommendations"""
    try:
        if not request.user_id or not request.movie_title:
            raise ValueError("Both user_id and movie_title are required for hybrid recommendations")
            
        recommendations = recommender.get_hybrid_recommendations(
            request.user_id,
            request.movie_title,
            request.n_recommendations
        )
        return {"recommendations": recommendations.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
