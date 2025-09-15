# eda_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

class EDAAnalysis:
    def __init__(self, file_path):
        """Configure EDA parser with data file path"""
        self.file_path = file_path
        self.df = None
        self.numeric_cols = ['price', 'rating', 'review_count']
        
    def load_data(self):
        """Download data from a CSV file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data uploaded successfully. Dimensions: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Data loading error: {e}")
            return False
    
    def basic_info(self):
        """Display basic information about the data"""
        print("=" * 50)
        print("Basic information about the data:")
        print("=" * 50)
        
        print(f"\nData dimensions: {self.df.shape}")
        print(f"\nNumber of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        
        print("\nInformation about data types:")
        print(self.df.dtypes)
        
        print("\nDescriptive statistics for numerical variables:")
        print(self.df.describe())
        
        print("\nmissing values:")
        missing_info = pd.DataFrame({
            'missing values': self.df.isnull().sum(),
            'percentage': (self.df.isnull().sum() / len(self.df)) * 100
        }).sort_values('missing values', ascending=False)
        
        print(missing_info.head(15))
    
    def univariate_analysis_numeric(self):
        """Univariate analysis of numerical variables"""
        print("\n" + "=" * 50)
        print("Univariate analysis of numerical variables:")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Price distribution
        sns.histplot(self.df['price'].dropna(), kde=True, ax=axes[0,0])
        axes[0,0].set_title('Price distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Price')
        axes[0,0].set_ylabel('Repetition')
        
        # Evaluation distribution
        sns.histplot(self.df['rating'].dropna(), kde=True, ax=axes[0,1])
        axes[0,1].set_title('Evaluation distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Evaluation')
        axes[0,1].set_ylabel('Repetition')
        
        # Distribution of the number of reviews (using a logarithmic scale)
        sns.histplot(np.log1p(self.df['review_count'].dropna()), kde=True, ax=axes[0,2])
        axes[0,2].set_title('Distribution of the number of reviews (logarithmic)', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Logarithm of number of reviews')
        axes[0,2].set_ylabel('Repetition')
        
        # Fund schemes
        sns.boxplot(y=self.df['price'], ax=axes[1,0])
        axes[1,0].set_title('Box diagram for Price', fontsize=14, fontweight='bold')
        
        sns.boxplot(y=self.df['rating'], ax=axes[1,1])
        axes[1,1].set_title('Box diagram for Rating', fontsize=14, fontweight='bold')
        
        sns.boxplot(y=np.log1p(self.df['review_count']), ax=axes[1,2])
        axes[1,2].set_title('Boxplot of number of reviews (logarithmic)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/univariate_numeric.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed statistics
        print("\nDetailed statistics for numerical variables:")
        for col in self.numeric_cols:
            print(f"\n{col}:")
            print(f"  Average: {self.df[col].mean():.2f}")
            print(f"  Median: {self.df[col].median():.2f}")
            print(f"  standard deviation: {self.df[col].std():.2f}")
            print(f"  range: {self.df[col].min():.2f} - {self.df[col].max():.2f}")
            print(f"  missing values: {self.df[col].isnull().sum()}")
    
    def univariate_analysis_categorical(self):
        """Univariate analysis of categorical variables"""
        print("\n" + "=" * 50)
        print("Univariate analysis of categorical variables:")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Brands (Top 10)
        top_brands = self.df['brand'].value_counts().head(10)
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=axes[0,0])
        axes[0,0].set_title('Top 10 Brands by Number of Products', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Number of products')
        axes[0,0].set_ylabel('Brand')
        
        # Categories (Top 15)
        top_categories = self.df['category_name'].value_counts().head(15)
        sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[0,1])
        axes[0,1].set_title('Top 15 categories by number of products', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Number of products')
        axes[0,1].set_ylabel('Category')
        
        # Distribution according to availability
        availability_counts = self.df['availability'].value_counts().head(5)
        sns.barplot(x=availability_counts.values, y=availability_counts.index, ax=axes[1,0])
        axes[1,0].set_title('Distribution of products according to availability', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Number of products')
        axes[1,0].set_ylabel('Availability status')
        
        # Availability pie chart
        availability_counts.plot.pie(autopct='%1.1f%%', ax=axes[1,1])
        axes[1,1].set_title('Availability rate', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('outputs/univariate_categorical.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Brand Statistics
        print("\nBrand Statistics:")
        brand_stats = self.df['brand'].value_counts()
        print(f"Number of unique brands: {brand_stats.nunique()}")
        print(f"Most popular brands:\n{brand_stats.head(10)}")
        
        # Category statistics
        print("\nCategory statistics:")
        category_stats = self.df['category_name'].value_counts()
        print(f"Number of unique categories: {category_stats.nunique()}")
        print(f"Most popular categories:\n{category_stats.head(10)}")
    
    def bivariate_analysis(self):
        """Bivariate Analysis"""
        print("\n" + "=" * 50)
        print("Bivariate Analysis:")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Price-rating relationship
        sns.scatterplot(data=self.df, x='price', y='rating', alpha=0.6, ax=axes[0,0])
        axes[0,0].set_title('The relationship between price and rating', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Price')
        axes[0,0].set_ylabel('Rating')
        
        # Price vs. Review Count
        sns.scatterplot(data=self.df, x='price', y='review_count', alpha=0.6, ax=axes[0,1])
        axes[0,1].set_title('Relationship between Price and Review Count', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Price')
        axes[0,1].set_ylabel('Review Count')
        axes[0,1].set_yscale('log')
        
        # Rating vs. Review Count
        sns.scatterplot(data=self.df, x='rating', y='review_count', alpha=0.6, ax=axes[1,0])
        axes[1,0].set_title('Relationship between Rating and Review Count', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Rating')
        axes[1,0].set_ylabel('Review Count')
        axes[1,0].set_yscale('log')
        
        # Correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[1,1])
        axes[1,1].set_title('Correlation matrix between numerical variables', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/bivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation coefficients
        print("\nCorrelation coefficients:")
        print(corr_matrix)
        
        # Statistical relationship testing
        print("\nStatistical relationship tests:")
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    corr, p_value = stats.pearsonr(self.df[col1].dropna(), self.df[col2].dropna())
                    print(f"{col1} vs {col2}: r = {corr:.3f}, p-value = {p_value:.3f}")
    
    def analyze_by_category(self):
        """Data analysis by categories"""
        print("\n" + "=" * 50)
        print("Data analysis by categories:")
        print("=" * 50)
        
        # Top 5 categories by number of products
        top_categories = self.df['category_name'].value_counts().head(5).index
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Average price by category
        price_by_category = self.df.groupby('category_name')['price'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=price_by_category.values, y=price_by_category.index, ax=axes[0,0])
        axes[0,0].set_title('Average price by category (top 10)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Average price')
        axes[0,0].set_ylabel('Category')
        
        # Average rating by category
        rating_by_category = self.df.groupby('category_name')['rating'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=rating_by_category.values, y=rating_by_category.index, ax=axes[0,1])
        axes[0,1].set_title('Average rating by category (top 10)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Average rating')
        axes[0,1].set_ylabel('Category')
        
        # Average review count by category
        reviews_by_category = self.df.groupby('category_name')['review_count'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=reviews_by_category.values, y=reviews_by_category.index, ax=axes[1,0])
        axes[1,0].set_title('Average review count by category (top 10)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Average review count')
        axes[1,0].set_ylabel('Category')
        axes[1,0].set_xscale('log')
        
        # Price distribution in the best categories
        for category in top_categories[:3]:
            category_data = self.df[self.df['category_name'] == category]['price']
            sns.kdeplot(category_data, label=category, ax=axes[1,1])
        
        axes[1,1].set_title('Price distribution in the top 3 categories', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Price')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/category_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed statistics by category
        print("\nDetailed statistics by category:")
        for category in top_categories:
            category_data = self.df[self.df['category_name'] == category]
            print(f"\nCategory: {category}")
            print(f"Number of products: {len(category_data)}")
            print(f"Average price: {category_data['price'].mean():.2f}")
            print(f"Average rating: {category_data['rating'].mean():.2f}")
            print(f"Average review count: {category_data['review_count'].mean():.2f}")
    
    def outlier_analysis(self):
        """Outlier analysis"""
        print("\n" + "=" * 50)
        print("Outlier analysis:")
        print("=" * 50)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Identifying outliers using IQR
        for i, col in enumerate(self.numeric_cols):
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            print(f"\nOutliers in {col}: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
            
            # Boxplot with outliers
            sns.boxplot(y=self.df[col], ax=axes[i])
            axes[i].set_title(f'Outliers in {col}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig('outputs/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def time_series_analysis(self):
        """Time series analysis (if there is a date column)"""
        print("\n" + "=" * 50)
        print("Time series analysis:")
        print("=" * 50)
        
        # Check for date columns
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            print(f"Date columns: {date_columns}")
            # Here you can add time series analysis.
        else:
            print("There are no date columns to parse.")
    
    def save_insights(self):
        """Save results and conclusions in a text file."""
        print("\n" + "=" * 50)
        print("Saving insights:")
        print("=" * 50)
        
        insights = []
        insights.append("Conclusions and insights from exploratory data analysis")
        insights.append("=" * 50)
        insights.append(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        insights.append(f"Number of products: {len(self.df)}")
        insights.append(f"Number of columns: {len(self.df.columns)}")
        
        # Conclusions about numerical variables
        insights.append("\n1. Numerical variables:")
        for col in self.numeric_cols:
            if col in self.df.columns:
                insights.append(f"   - {col}: mean = {self.df[col].mean():.2f}, median = {self.df[col].median():.2f}")
        
        # Conclusions about missing values
        missing_cols = self.df.isnull().sum().sort_values(ascending=False).head(5)
        insights.append("\n2. Missing values:")
        for col, count in missing_cols.items():
            if count > 0:
                insights.append(f"   - {col}: {count} missing values ({(count/len(self.df))*100:.1f}%)")
        
        # Conclusions about relationships
        insights.append("\n3. Relationships between variables:")
        corr_matrix = self.df[self.numeric_cols].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                insights.append(f"   - {col1} vs {col2}: correlation = {corr_value:.3f}")
        
        # Save file
        with open('outputs/data_insights.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights))
        
        print("The conclusions were saved in a file. 'outputs/data_insights.txt'")
    
    def run_full_analysis(self):
        """Run full analysis"""
        print("Start comprehensive exploratory analysis...")
        
        if not self.load_data():
            return
        
        # Create an output folder
        import os
        os.makedirs('outputs', exist_ok=True)
        
        # Run all analysis
        self.basic_info()
        self.univariate_analysis_numeric()
        self.univariate_analysis_categorical()
        self.bivariate_analysis()
        self.analyze_by_category()
        self.outlier_analysis()
        self.time_series_analysis()
        self.save_insights()
        
        print("\n" + "=" * 50)
        print("The comprehensive exploratory analysis has been completed.!")
        print("=" * 50)

# Main execution code
if __name__ == "__main__":
    # Data preparation and analysis
    analyzer = EDAAnalysis('data/processed/cleaned_amazon_data.csv')
    analyzer.run_full_analysis()
    
    # Show some examples of data
    print("\nSample data:")
    print(analyzer.df.head())
    
    print("\nAnalysis is ready! All outputs have been saved in a folder. 'outputs'")