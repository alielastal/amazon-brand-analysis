# -*- coding: utf-8 -*-
"""
Amazon Electronics Recommendation API
Deployment script for the recommendation system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load recommendation system assets
try:
    with open('/content/drive/MyDrive/amazon-sentiment-analysis/recommendation_system.pkl', 'rb') as f:
        assets = pickle.load(f)
    
    feature_matrix = assets['feature_matrix']
    brand_encoder = assets['brand_encoder']
    category_encoder = assets['category_encoder']
    scaler = assets['scaler']
    df = assets['df']
    
    logger.info("✅ Recommendation system loaded successfully!")
    logger.info(f"📊 Loaded {len(df)} products and {len(brand_encoder)} brands")
    
except Exception as e:
    logger.error(f"❌ Failed to load recommendation system: {e}")
    raise e

def content_based_recommendation(product_id, n_recommendations=5):
    """Content-based recommendation function"""
    try:
        product_idx = df[df['id'] == product_id].index[0]
        similarity_matrix = cosine_similarity(feature_matrix)
        similar_indices = similarity_matrix[product_idx].argsort()[::-1][1:n_recommendations+1]
        
        recommendations = df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = similarity_matrix[product_idx][similar_indices]
        
        return recommendations.to_dict('records')
    except Exception as e:
        logger.error(f"Content-based recommendation failed: {e}")
        return []

def user_preference_recommendation(preferences, n_recommendations=10):
    """User preference-based recommendation function"""
    try:
        filtered_df = df.copy()
        
        # Apply filters based on user preferences
        if 'brands' in preferences and preferences['brands']:
            filtered_df = filtered_df[filtered_df['brand'].isin(preferences['brands'])]
        
        if 'max_price' in preferences:
            filtered_df = filtered_df[filtered_df['price'] <= preferences['max_price']]
        
        if 'min_rating' in preferences:
            filtered_df = filtered_df[filtered_df['rating'] >= preferences['min_rating']]
        
        if 'category' in preferences and preferences['category']:
            filtered_df = filtered_df[filtered_df['category_name'] == preferences['category']]
        
        if len(filtered_df) == 0:
            return []
        
        # Calculate recommendation score
        filtered_df['recommendation_score'] = (filtered_df['rating'] * filtered_df['review_count']) / filtered_df['price']
        
        # Return top recommendations
        top_recommendations = filtered_df.nlargest(n_recommendations, 'recommendation_score')
        return top_recommendations.to_dict('records')
        
    except Exception as e:
        logger.error(f"User preference recommendation failed: {e}")
        return []

def brand_strategy_recommendation(strategy_type, n_recommendations=5):
    """Brand strategy-based recommendation function"""
    try:
        if strategy_type == 'premium':
            filtered_df = df[df['rating'] >= 4.0]
            filtered_df = filtered_df.nlargest(n_recommendations * 2, 'price')
            return filtered_df.nlargest(n_recommendations, 'rating').to_dict('records')
        
        elif strategy_type == 'value':
            filtered_df = df[df['rating'] >= 4.0]
            filtered_df['value_score'] = filtered_df['rating'] * filtered_df['review_count'] / filtered_df['price']
            return filtered_df.nlargest(n_recommendations, 'value_score').to_dict('records')
        
        elif strategy_type == 'budget':
            filtered_df = df[df['price'] <= df['price'].quantile(0.33)]
            return filtered_df.nlargest(n_recommendations, 'rating').to_dict('records')
        
        else:
            return []
            
    except Exception as e:
        logger.error(f"Brand strategy recommendation failed: {e}")
        return []

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Amazon Electronics Recommendation API',
        'version': '1.0',
        'endpoints': {
            '/recommend/content/<product_id>': 'Content-based recommendations',
            '/recommend/preferences': 'User preference recommendations (POST)',
            '/recommend/strategy/<strategy_type>': 'Strategy-based recommendations',
            '/products': 'Get all products',
            '/brands': 'Get all brands',
            '/categories': 'Get all categories'
        }
    })

@app.route('/recommend/content/<product_id>', methods=['GET'])
def recommend_content(product_id):
    """Content-based recommendation endpoint"""
    try:
        n_recommendations = int(request.args.get('n', 5))
        recommendations = content_based_recommendation(product_id, n_recommendations)
        
        return jsonify({
            'success': True,
            'product_id': product_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/recommend/preferences', methods=['POST'])
def recommend_preferences():
    """User preference recommendation endpoint"""
    try:
        data = request.get_json()
        n_recommendations = int(request.args.get('n', 10))
        
        recommendations = user_preference_recommendation(data, n_recommendations)
        
        return jsonify({
            'success': True,
            'preferences': data,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/recommend/strategy/<strategy_type>', methods=['GET'])
def recommend_strategy(strategy_type):
    """Strategy-based recommendation endpoint"""
    try:
        if strategy_type not in ['premium', 'value', 'budget']:
            return jsonify({
                'success': False,
                'error': 'Strategy must be: premium, value, or budget'
            }), 400
        
        n_recommendations = int(request.args.get('n', 5))
        recommendations = brand_strategy_recommendation(strategy_type, n_recommendations)
        
        return jsonify({
            'success': True,
            'strategy': strategy_type,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/products', methods=['GET'])
def get_products():
    """Get all products endpoint"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        products = df.iloc[start_idx:end_idx].to_dict('records')
        
        return jsonify({
            'success': True,
            'page': page,
            'per_page': per_page,
            'total_products': len(df),
            'products': products
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/brands', methods=['GET'])
def get_brands():
    """Get all brands endpoint"""
    try:
        brands = sorted(df['brand'].unique().tolist())
        return jsonify({
            'success': True,
            'brands': brands,
            'count': len(brands)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get all categories endpoint"""
    try:
        categories = sorted(df['category_name'].unique().tolist())
        return jsonify({
            'success': True,
            'categories': categories,
            'count': len(categories)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Amazon Recommendation API...")
    logger.info("📚 Available endpoints:")
    logger.info("   GET  /recommend/content/<product_id>")
    logger.info("   POST /recommend/preferences")
    logger.info("   GET  /recommend/strategy/<strategy_type>")
    logger.info("   GET  /products")
    logger.info("   GET  /brands")
    logger.info("   GET  /categories")
    
    app.run(host='0.0.0.0', port=5000, debug=True)