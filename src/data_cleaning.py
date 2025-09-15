import pandas as pd
import numpy as np
from scipy import stats
import logging
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config_path='config.yaml'):
        """
        Initialize DataCleaner with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.df = None
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'data_cleaning': {
                    'missing_threshold': 0.5,
                    'z_score_threshold': 3
                }
            }
    
    def load_data(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataframe
        """
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def initial_inspection(self):
        """Perform initial data inspection"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Performing initial data inspection...")
        
        # Basic info
        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        logger.info("Missing values per column:")
        for col, count, percentage in zip(missing_values.index, missing_values.values, missing_percentage.values):
            if count > 0:
                logger.info(f"  {col}: {count} ({percentage:.2f}%)")
        
        return missing_values
    
    def remove_high_missing_columns(self, threshold=None):
        """
        Remove columns with high percentage of missing values
        
        Args:
            threshold (float): Missing percentage threshold (0-1)
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        if threshold is None:
            threshold = self.config.get('data_cleaning', {}).get('missing_threshold', 0.5)
        
        missing_percentage = self.df.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > threshold].index
        
        if len(columns_to_drop) > 0:
            logger.info(f"Dropping {len(columns_to_drop)} columns with >{threshold*100}% missing values")
            self.df = self.df.drop(columns=columns_to_drop)
        
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Impute numerical columns with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed {col} with median: {median_val}")
        
        # Remove rows with missing critical columns
        critical_cols = ['review_count']
        for col in critical_cols:
            if col in self.df.columns:
                initial_count = len(self.df)
                self.df = self.df.dropna(subset=[col])
                removed_count = initial_count - len(self.df)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} rows with missing {col}")
        
        return self.df
    
    def detect_outliers(self, method='zscore', threshold=None):
        """
        Detect outliers in numerical columns
        
        Args:
            method (str): 'zscore' or 'iqr'
            threshold (float): Threshold for outlier detection
            
        Returns:
            dict: Outliers information
        """
        if threshold is None:
            threshold = self.config.get('data_cleaning', {}).get('z_score_threshold', 3)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numerical_cols:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = self.df[z_scores > threshold]
            elif method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            outliers_info[col] = {
                'count': len(outliers),
                'indices': outliers.index.tolist(),
                'values': outliers[col].tolist()
            }
            
            logger.info(f"Outliers in {col}: {len(outliers)}")
        
        return outliers_info
    
    def remove_outliers(self, outliers_info):
        """
        Remove outliers based on detected outliers information
        
        Args:
            outliers_info (dict): Outliers information from detect_outliers()
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        indices_to_remove = set()
        
        for col, info in outliers_info.items():
            indices_to_remove.update(info['indices'])
        
        initial_count = len(self.df)
        self.df = self.df.drop(index=list(indices_to_remove))
        removed_count = initial_count - len(self.df)
        
        logger.info(f"Removed {removed_count} outlier rows")
        return self.df
    
    def clean_data(self, file_path, save_path=None):
        """
        Complete data cleaning pipeline
        
        Args:
            file_path (str): Path to input CSV file
            save_path (str): Path to save cleaned data (optional)
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        try:
            # Load data
            self.load_data(file_path)
            
            # Initial inspection
            self.initial_inspection()
            
            # Remove high missing columns
            self.remove_high_missing_columns()
            
            # Handle missing values
            self.handle_missing_values()
            
            # Detect and remove outliers
            outliers_info = self.detect_outliers(method='zscore')
            self.remove_outliers(outliers_info)
            
            # Save cleaned data
            if save_path:
                self.save_cleaned_data(save_path)
                logger.info(f"Cleaned data saved to: {save_path}")
            
            logger.info("Data cleaning completed successfully!")
            return self.df
            
        except Exception as e:
            logger.error(f"Error in data cleaning pipeline: {str(e)}")
            raise
    
    def save_cleaned_data(self, save_path):
        """Save cleaned data to CSV file"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    
    # Clean data
    cleaned_df = cleaner.clean_data(
        file_path='data/raw/amazon_electronics.csv',
        save_path='data/processed/cleaned_amazon_data.csv'
    )
    
    print(f"Final dataset shape: {cleaned_df.shape}")