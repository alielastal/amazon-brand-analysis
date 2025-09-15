# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os
import uvicorn
import json

# Create FastAPI app
app = FastAPI(
    title="Amazon Recommendation API",
    description="API for product recommendations and analytics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded data
df = None
feature_matrix = None
brand_encoder = None
category_encoder = None
scaler = None

def clean_dataframe_for_json(df):
    """Clean DataFrame to ensure all values are JSON serializable"""
    df_clean = df.copy()
    
    # Replace NaN with None (which becomes null in JSON)
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    # Replace infinite values with None
    for col in df_clean.columns:
        if df_clean[col].dtype == np.float64 or df_clean[col].dtype == np.float32:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], None)
    
    return df_clean

def load_data():
    """Load data and models when the API starts"""
    global df, feature_matrix, brand_encoder, category_encoder, scaler
    
    try:
        # Load the cleaned data
        data_path = "data/processed/cleaned_amazon_data.csv"
        df = pd.read_csv(data_path)
        
        # Clean the data for JSON serialization
        df = clean_dataframe_for_json(df)
        
        # Try to load precomputed recommendation assets
        model_path = "models/recommendation_system.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                assets = pickle.load(f)
                feature_matrix = assets['feature_matrix']
                brand_encoder = assets['brand_encoder']
                category_encoder = assets['category_encoder']
                scaler = assets['scaler']
        else:
            # If no precomputed assets, compute them
            initialize_recommendation_system()
            
        print("✅ Data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def initialize_recommendation_system():
    """Initialize the recommendation system components"""
    global df, feature_matrix, brand_encoder, category_encoder, scaler
    
    # Create recommendation features
    rec_features = df[['brand', 'category_name', 'price', 'rating', 'review_count']].copy()

    # Handle missing values
    rec_features = rec_features.fillna({
        'rating': df['rating'].median(),
        'review_count': df['review_count'].median(),
        'category_name': 'Unknown',
        'brand': 'Unknown'
    })

    # Encode categorical variables
    brand_encoder = {brand: idx for idx, brand in enumerate(rec_features['brand'].unique())}
    category_encoder = {cat: idx for idx, cat in enumerate(rec_features['category_name'].unique())}

    rec_features['brand_encoded'] = rec_features['brand'].map(brand_encoder)
    rec_features['category_encoded'] = rec_features['category_name'].map(category_encoder)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['price', 'rating', 'review_count']
    rec_features[numerical_features] = scaler.fit_transform(rec_features[numerical_features])

    # Create feature matrix
    feature_columns = ['brand_encoded', 'category_encoded', 'price', 'rating', 'review_count']
    feature_matrix = rec_features[feature_columns].values

def content_based_recommendation(product_id: int, n_recommendations: int = 5):
    """Recommend similar products based on content features"""
    if product_id >= len(df) or product_id < 0:
        raise HTTPException(status_code=404, detail="Product ID not found")
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(feature_matrix)

    # Get similar products
    similar_indices = similarity_matrix[product_id].argsort()[::-1][1:n_recommendations+1]

    # Get recommendations
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = similarity_matrix[product_id][similar_indices]
    
    # Clean for JSON serialization
    recommendations = clean_dataframe_for_json(recommendations)
    
    return recommendations

def user_preference_recommendation(user_preferences: Dict[str, Any], n_recommendations: int = 10):
    """Recommend products based on user preferences"""
    filtered_df = df.copy()

    if 'brands' in user_preferences and user_preferences['brands']:
        filtered_df = filtered_df[filtered_df['brand'].isin(user_preferences['brands'])]

    if 'max_price' in user_preferences:
        filtered_df = filtered_df[filtered_df['price'] <= user_preferences['max_price']]

    if 'min_rating' in user_preferences:
        filtered_df = filtered_df[filtered_df['rating'] >= user_preferences['min_rating']]

    if 'categories' in user_preferences and user_preferences['categories']:
        filtered_df = filtered_df[filtered_df['category_name'].isin(user_preferences['categories'])]

    if len(filtered_df) == 0:
        return pd.DataFrame()  # No products match preferences

    # Score products based on quality and value
    filtered_df['recommendation_score'] = (filtered_df['rating'] * filtered_df['review_count']) / filtered_df['price']

    # Return top recommendations
    recommendations = filtered_df.nlargest(n_recommendations, 'recommendation_score')
    
    # Clean for JSON serialization
    recommendations = clean_dataframe_for_json(recommendations)
    
    return recommendations

def brand_strategy_recommendation(strategy_type: str, n_recommendations: int = 5):
    """Recommend products based on brand strategy"""
    if strategy_type == 'premium':
        # Premium quality: high rating, higher price
        filtered_df = df[df['rating'] >= 4.0]
        filtered_df = filtered_df.nlargest(n_recommendations * 2, 'price')
        recommendations = filtered_df.nlargest(n_recommendations, 'rating')

    elif strategy_type == 'value':
        # Best value: high rating, reasonable price
        filtered_df = df[df['rating'] >= 4.0]
        filtered_df['value_score'] = filtered_df['rating'] * filtered_df['review_count'] / filtered_df['price']
        recommendations = filtered_df.nlargest(n_recommendations, 'value_score')

    elif strategy_type == 'budget':
        # Budget: good rating, low price
        filtered_df = df[df['price'] <= df['price'].quantile(0.33)]
        recommendations = filtered_df.nlargest(n_recommendations, 'rating')

    else:
        recommendations = pd.DataFrame()
    
    # Clean for JSON serialization
    recommendations = clean_dataframe_for_json(recommendations)
    
    return recommendations

# Load data when the app starts
@app.on_event("startup")
async def startup_event():
    load_data()

@app.get("/")
async def root():
    return {
        "message": "Amazon Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "recommendations": "/api/v1/recommendations",
            "products": "/api/v1/products",
            "brands": "/api/v1/brands",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "data_loaded": df is not None,
        "products_count": len(df) if df is not None else 0
    }

@app.get("/api/v1/products")
async def get_products(
    skip: int = 0, 
    limit: int = 10,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None
):
    """Get products with filtering options"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    filtered_df = df.copy()
    
    if brand:
        filtered_df = filtered_df[filtered_df['brand'] == brand]
    if category:
        filtered_df = filtered_df[filtered_df['category_name'] == category]
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['price'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['price'] <= max_price]
    if min_rating is not None:
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    # Apply pagination
    result_df = filtered_df.iloc[skip:skip + limit]
    
    # Clean for JSON serialization
    result_df = clean_dataframe_for_json(result_df)
    
    return {
        "total": len(filtered_df),
        "skip": skip,
        "limit": limit,
        "products": result_df.to_dict(orient='records')
    }

@app.get("/api/v1/products/{product_id}")
async def get_product(product_id: int):
    """Get a specific product by ID"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if product_id < 0 or product_id >= len(df):
        raise HTTPException(status_code=404, detail="Product not found")
    
    product_data = df.iloc[product_id].to_dict()
    
    # Clean for JSON serialization
    for key, value in product_data.items():
        if pd.isna(value) or value in [np.inf, -np.inf]:
            product_data[key] = None
    
    return product_data

@app.get("/api/v1/brands")
async def get_brands():
    """Get list of all brands"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    brands = [brand for brand in df['brand'].unique() if brand is not None and not pd.isna(brand)]
    return {"brands": brands}

@app.get("/api/v1/categories")
async def get_categories():
    """Get list of all categories"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    categories = [cat for cat in df['category_name'].unique() if cat is not None and not pd.isna(cat)]
    return {"categories": categories}

@app.get("/api/v1/recommendations/content-based/{product_id}")
async def get_content_based_recommendations(
    product_id: int, 
    limit: int = Query(5, ge=1, le=20)
):
    """Get content-based recommendations for a product"""
    if df is None or feature_matrix is None:
        raise HTTPException(status_code=503, detail="Recommendation system not ready")
    
    recommendations = content_based_recommendation(product_id, limit)
    
    # Get original product data
    original_product = df.iloc[product_id].to_dict()
    for key, value in original_product.items():
        if pd.isna(value) or value in [np.inf, -np.inf]:
            original_product[key] = None
    
    return {
        "original_product": original_product,
        "recommendations": recommendations.to_dict(orient='records')
    }

@app.post("/api/v1/recommendations/user-based")
async def get_user_based_recommendations(
    preferences: Dict[str, Any],
    limit: int = Query(10, ge=1, le=20)
):
    """Get recommendations based on user preferences"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    recommendations = user_preference_recommendation(preferences, limit)
    
    if len(recommendations) == 0:
        raise HTTPException(status_code=404, detail="No products match your preferences")
    
    return {
        "preferences": preferences,
        "recommendations": recommendations.to_dict(orient='records')
    }

@app.get("/api/v1/recommendations/strategy-based/{strategy_type}")
async def get_strategy_based_recommendations(
    strategy_type: str,
    limit: int = Query(5, ge=1, le=20)
):
    """Get recommendations based on strategy (premium, value, budget)"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if strategy_type not in ['premium', 'value', 'budget']:
        raise HTTPException(status_code=400, detail="Strategy must be 'premium', 'value', or 'budget'")
    
    recommendations = brand_strategy_recommendation(strategy_type, limit)
    
    return {
        "strategy": strategy_type,
        "recommendations": recommendations.to_dict(orient='records')
    }

@app.get("/api/v1/stats/price-distribution")
async def get_price_distribution():
    """Get price distribution statistics"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Filter out NaN and infinite values
    price_data = df['price'].dropna()
    price_data = price_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    return {
        "min_price": float(price_data.min()) if len(price_data) > 0 else 0,
        "max_price": float(price_data.max()) if len(price_data) > 0 else 0,
        "mean_price": float(price_data.mean()) if len(price_data) > 0 else 0,
        "median_price": float(price_data.median()) if len(price_data) > 0 else 0
    }

@app.get("/api/v1/stats/brand-stats")
async def get_brand_stats():
    """Get statistics by brand"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Clean data before aggregation
    clean_df = df.copy()
    clean_df['price'] = clean_df['price'].replace([np.inf, -np.inf], np.nan)
    clean_df['rating'] = clean_df['rating'].replace([np.inf, -np.inf], np.nan)
    
    brand_stats = clean_df.groupby('brand').agg({
        'price': ['mean', 'count'],
        'rating': 'mean'
    }).round(2)
    
    brand_stats.columns = ['avg_price', 'product_count', 'avg_rating']
    
    # Clean the result for JSON
    result = brand_stats.reset_index()
    result = clean_dataframe_for_json(result)
    
    return result.to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)