# recommendation_system.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")
print("✅ Libraries imported successfully!")

def content_based_recommendation(product_idx, feature_matrix, df, n_recommendations=5):
    """
    Recommend similar products based on content features
    """
    # Calculate similarity
    similarity_matrix = cosine_similarity(feature_matrix)

    # Get similar products
    similar_indices = similarity_matrix[product_idx].argsort()[::-1][1:n_recommendations+1]

    # Get recommendations
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = similarity_matrix[product_idx][similar_indices]

    return recommendations

def user_preference_recommendation(user_preferences, df, n_recommendations=10):
    """
    Recommend products based on user preferences
    user_preferences: dict with {'brand': [], 'max_price': x, 'min_rating': y}
    """
    # Filter based on preferences
    filtered_df = df.copy()

    if 'brand' in user_preferences and user_preferences['brand']:
        filtered_df = filtered_df[filtered_df['brand'].isin(user_preferences['brand'])]

    if 'max_price' in user_preferences:
        filtered_df = filtered_df[filtered_df['price'] <= user_preferences['max_price']]

    if 'min_rating' in user_preferences:
        filtered_df = filtered_df[filtered_df['rating'] >= user_preferences['min_rating']]

    if len(filtered_df) == 0:
        return pd.DataFrame()  # No products match preferences

    # Score products based on quality and value
    filtered_df['recommendation_score'] = (filtered_df['rating'] * filtered_df['review_count']) / filtered_df['price']

    # Return top recommendations
    return filtered_df.nlargest(n_recommendations, 'recommendation_score')

def brand_strategy_recommendation(strategy_type, df, n_recommendations=5):
    """
    Recommend products based on brand strategy
    strategy_type: 'premium', 'value', 'budget'
    """
    if strategy_type == 'premium':
        # Premium quality: high rating, higher price
        filtered_df = df[df['rating'] >= 4.0]
        filtered_df = filtered_df.nlargest(n_recommendations * 2, 'price')
        return filtered_df.nlargest(n_recommendations, 'rating')

    elif strategy_type == 'value':
        # Best value: high rating, reasonable price
        filtered_df = df[df['rating'] >= 4.0]
        filtered_df['value_score'] = filtered_df['rating'] * filtered_df['review_count'] / filtered_df['price']
        return filtered_df.nlargest(n_recommendations, 'value_score')

    elif strategy_type == 'budget':
        # Budget: good rating, low price
        filtered_df = df[df['price'] <= df['price'].quantile(0.33)]
        return filtered_df.nlargest(n_recommendations, 'rating')

    else:
        return pd.DataFrame()

def evaluate_recommendation_quality(recommendations, original_product):
    """
    Evaluate how good recommendations are
    """
    if len(recommendations) == 0:
        return 0

    # Score based on similarity in key metrics
    price_similarity = 1 - abs(recommendations['price'].mean() - original_product['price']) / original_product['price']
    rating_similarity = 1 - abs(recommendations['rating'].mean() - original_product['rating']) / 5
    brand_match = (recommendations['brand'] == original_product['brand']).mean()

    total_score = (price_similarity + rating_similarity + brand_match) / 3
    return total_score

def main():
    """Main function to run the recommendation system pipeline."""
    
    # Set working directory and file path
    file_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(file_dir)
    file_path = os.path.join(script_dir, "data/processed/cleaned_amazon_data.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("Please make sure 'cleaned_amazon_data.csv' is in the same directory as this script.")
        return
    
    # Load cleaned data
    df = pd.read_csv(file_path)
    
    # Create recommendation features
    rec_features = df[['brand', 'category_name', 'price', 'rating', 'review_count']].copy()

    # Handle missing values
    rec_features = rec_features.fillna({
        'rating': df['rating'].median(),
        'review_count': df['review_count'].median(),
        'category_name': 'Unknown',
        'brand': 'Unknown'
    })

    print("📊 Recommendation dataset prepared!")
    print(f"📈 Total products: {len(rec_features)}")
    
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

    print("✅ Feature matrix prepared for recommendation!")
    print(f"📊 Feature matrix shape: {feature_matrix.shape}")
    
    # Test the content-based system
    test_product_idx = 0  # First product
    recommendations = content_based_recommendation(test_product_idx, feature_matrix, df)

    print("🎯 RECOMMENDATION SYSTEM TEST:")
    print("=" * 50)
    print(f"Original Product: {df['title'].iloc[test_product_idx][:50]}...")
    print(f"Brand: {df['brand'].iloc[test_product_idx]}")
    print(f"Price: ${df['price'].iloc[test_product_idx]:.2f}")
    print(f"Rating: {df['rating'].iloc[test_product_idx]}⭐")
    print("\n📋 Top 5 Recommendations:")
    print("=" * 30)
    for idx, row in recommendations.iterrows():
        print(f"• {row['title'][:40]}... | {row['brand']} | ${row['price']:.2f} | {row['rating']}⭐ | Similarity: {row['similarity_score']:.3f}")
    
    # Test user preferences
    user_prefs = {
        'brand': ['HP', 'JJC', 'Alestor'],
        'max_price': 100,
        'min_rating': 4.0
    }

    user_recommendations = user_preference_recommendation(user_prefs, df)

    print("\n🎯 USER PREFERENCE RECOMMENDATIONS:")
    print("=" * 50)
    print(f"Preferences: Brands {user_prefs['brand']}, Max Price ${user_prefs['max_price']}, Min Rating {user_prefs['min_rating']}⭐")
    print(f"Found {len(user_recommendations)} matching products")
    print("\n📋 Top Recommendations:")
    print("=" * 30)
    for idx, row in user_recommendations.head().iterrows():
        print(f"• {row['title'][:35]}... | {row['brand']} | ${row['price']:.2f} | {row['rating']}⭐ | Score: {row['recommendation_score']:.2f}")
    
    # Test different strategies
    strategies = ['premium', 'value', 'budget']

    print("\n🎯 BRAND STRATEGY RECOMMENDATIONS:")
    print("=" * 50)

    for strategy in strategies:
        recs = brand_strategy_recommendation(strategy, df)
        print(f"\n📋 {strategy.upper()} STRATEGY ({len(recs)} products):")
        print("-" * 30)
        for idx, row in recs.iterrows():
            print(f"• {row['title'][:30]}... | {row['brand']} | ${row['price']:.2f} | {row['rating']}⭐")
    
    # Test evaluation
    test_product = df.iloc[0]
    recs = content_based_recommendation(0, feature_matrix, df)
    quality_score = evaluate_recommendation_quality(recs, test_product)

    print("\n📊 RECOMMENDATION SYSTEM EVALUATION:")
    print("=" * 50)
    print(f"Original Product: {test_product['brand']} - ${test_product['price']:.2f} - {test_product['rating']}⭐")
    print(f"Recommendation Quality Score: {quality_score:.3f}/1.0")
    print(f"Average Price of Recommendations: ${recs['price'].mean():.2f}")
    print(f"Average Rating of Recommendations: {recs['rating'].mean():.1f}⭐")
    
    # Save the recommendation system components
    recommendation_assets = {
        'feature_matrix': feature_matrix,
        'brand_encoder': brand_encoder,
        'category_encoder': category_encoder,
        'scaler': scaler,
        'df': df
    }

    output_path = os.path.join(script_dir, 'models/recommendation_system.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(recommendation_assets, f)

    print(f"\n✅ Recommendation system saved to {output_path}!")
    print("🎯 System includes: Content-based, User-preference, and Strategy-based recommendations")
    print("📊 Ready for deployment in web application!")
    
    # Final summary
    print("\n🎯 RECOMMENDATION SYSTEM DEMO COMPLETED!")
    print("=" * 50)
    print("📋 System Capabilities:")
    print("• Content-based similarity recommendations")
    print("• User preference filtering (brand, price, rating)")
    print("• Brand strategy-based recommendations (Premium/Value/Budget)")
    print("• Quality evaluation metrics")
    print(f"• Covers {len(df)} products across {len(brand_encoder)} brands")

    print("\n💡 Business Insights:")
    print("• HP products are great for premium recommendations")
    print("• JJC offers best value for money")
    print("• System can personalize recommendations based on user budget and preferences")

    print("\n✅ Ready for integration with web application!")

if __name__ == "__main__":
    main()