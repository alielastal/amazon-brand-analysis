# brand_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setting up basic chart formats
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

def load_data(file_path):
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        print("✅ Data uploaded successfully!")
        print(f"📊 Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ The file was not found. Please check the correct path..")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading data.: {e}")
        return None

def basic_brand_overview(df):
    """Basic Overview of Brands"""
    brand_stats = df['brand'].value_counts()
    print("🏷️ Brand Distribution:")
    print(f"Total Brands: {len(brand_stats)}")
    print(f"Top 10 Brands That Represent {brand_stats.head(10).sum()/len(df)*100:.1f}% from the products")
    
    # Top 15 Brands Chart
    plt.figure(figsize=(14, 8))
    brand_stats.head(15).plot(kind='bar', color='lightblue', edgecolor='black')
    plt.title('Top 15 Brands by Number of Products', fontsize=16, fontweight='bold')
    plt.xlabel('Brand', fontsize=12)
    plt.ylabel('Number of products', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/top_15_brands.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return brand_stats

def price_analysis_by_brand(df):
    """Price Analysis by Brand"""
    brand_price_analysis = df.groupby('brand').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'rating': 'mean',
        'review_count': 'sum'
    }).round(2)
    
    brand_price_analysis.columns = ['avg_price', 'median_price', 'price_std', 'product_count', 'avg_rating', 'total_reviews']
    brand_price_analysis = brand_price_analysis.sort_values('avg_price', ascending=False)
    
    print("📈 Top 10 Most Expensive Brands:")
    print(brand_price_analysis.head(10))
    print("\n📉 10 Cheapest Brands:")
    print(brand_price_analysis.tail(10))
    
    return brand_price_analysis

def visualize_price_vs_rating(brand_price_analysis):
    """Visualize the relationship between price and valuation by brand."""
    plt.figure(figsize=(14, 8))
    
    # Filter brands that have more than 5 products
    top_brands = brand_price_analysis[brand_price_analysis['product_count'] > 5].head(20)
    
    scatter = plt.scatter(top_brands['avg_price'], top_brands['avg_rating'],
                         s=top_brands['product_count']*10,
                         c=top_brands['total_reviews'],
                         alpha=0.6, cmap='viridis')
    
    plt.colorbar(scatter, label='Total Reviews')
    plt.xlabel('Average price ($)', fontsize=12)
    plt.ylabel('Average rating', fontsize=12)
    plt.title('Brand Analysis: Price vs. Valuation (Volume = Number of Products)', 
              fontsize=16, fontweight='bold')
    
    # Add brand names
    for i, brand in enumerate(top_brands.index):
        plt.annotate(brand, (top_brands['avg_price'].iloc[i], top_brands['avg_rating'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/price_vs_rating_by_brand.png', dpi=300, bbox_inches='tight')
    plt.show()

def advanced_brand_analysis(df, brand_price_analysis):
    """Advanced Brand Analysis"""
    # Top-rated brands with more than 5 products
    high_rated_brands = brand_price_analysis[brand_price_analysis['product_count'] > 5]
    high_rated_brands = high_rated_brands.sort_values('avg_rating', ascending=False)
    
    print("\n🏆 Top 10 Rated Brands (with more than 5 products):")
    print(high_rated_brands.head(10)[['avg_rating', 'product_count', 'avg_price']])
    
    # Most Reviewed Brands
    most_reviewed_brands = brand_price_analysis.sort_values('total_reviews', ascending=False)
    
    print("\n📝 Most Reviewed Brands:")
    print(most_reviewed_brands.head(10)[['total_reviews', 'avg_rating', 'avg_price']])
    
    # Calculating the relationship between price and rating
    correlation = df[['price', 'rating']].corr().iloc[0, 1]
    print(f"\n📊 Price-Rating Correlation Coefficient: {correlation:.3f}")

def main():
    """Main function"""
    # Loading Data
    file_path = "data/processed/cleaned_amazon_data.csv"
    df = load_data(file_path)
    
    if df is None:
        return
    
    # Fundamental analysis
    print("\n" + "="*50)
    print("Fundamental Brand Analysis")
    print("="*50)
    brand_stats = basic_brand_overview(df)
    
    # Price analysis
    print("\n" + "="*50)
    print("Price Analysis by Brand")
    print("="*50)
    brand_price_analysis = price_analysis_by_brand(df)
    
    # Visualizing
    print("\n" + "="*50)
    print("Visualize the relationship between price and rating")
    print("="*50)
    visualize_price_vs_rating(brand_price_analysis)
    
    # Advanced Analysis
    print("\n" + "="*50)
    print("Advanced Brand Analysis")
    print("="*50)
    advanced_brand_analysis(df, brand_price_analysis)
    
    # Save results to a file
    try:
        brand_price_analysis.to_csv('outputs/brand_analysis_results.csv')
        print("\n💾 The results are saved in a file 'brand_analysis_results.csv'")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()