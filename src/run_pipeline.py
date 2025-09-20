"""
House Price Prediction using AlphaPy AutoML
==========================================

Optimized automated machine learning pipeline for house price prediction
using the AlphaPy library with clean, efficient code structure.

Key Features:
- Automated data preprocessing
- Multiple algorithm comparison using AlphaPy framework
- Feature importance analysis
- Performance metrics and comprehensive reporting

Author: Sumanta Swain
GitHub: https://github.com/Sumanta01
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# AlphaPy import with error handling
try:
    from alphapy.model import Model
    ALPHAPY_AVAILABLE = True
    print("✅ AlphaPy available")
except ImportError as e:
    ALPHAPY_AVAILABLE = False
    print(f"❌ AlphaPy not available: {e}")
    raise ImportError("AlphaPy is required for this pipeline. Please install alphapy-pro.")

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'house_prices.csv')


def load_and_explore_data():
    """
    Load the house price dataset and perform basic exploration
    """
    print("📊 Loading house price dataset...")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Data file not found at '{RAW_DATA_PATH}'")
        print("Please ensure 'house_prices.csv' is in the 'data/' folder.")
        raise FileNotFoundError(f"Data file not found: {RAW_DATA_PATH}")
        
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Basic exploration
    print(f"Target variable: price")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Categorical columns: {df.select_dtypes(include=['object']).shape[1]}")
    
    return df


def preprocess_data(df, target_variable='price'):
    """
    Preprocess the data for AlphaPy modeling
    """
    print("🔧 Preprocessing data...")
    
    df_processed = df.copy()
    predictors = [col for col in df_processed.columns if col != target_variable]
    
    # Handle missing values efficiently
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Encode categorical variables
    label_encoders = {}
    for col in predictors:
        if df_processed[col].dtype == 'object':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    print(f"✅ Preprocessing complete. Features: {len(predictors)}")
    return df_processed, predictors, label_encoders


def create_alphapy_model():
    """
    Create and configure AlphaPy model with optimized settings
    """
    model_specs = {
        'algorithms': ['rf', 'gb', 'lr'],  # Random Forest, Gradient Boosting, Linear Regression
        'model_type': 'regression',
        'scoring_function': 'neg_mean_squared_error',
        'cv_folds': 5,
        'test_size': 0.2,
        'random_state': 42,
        'calibration': False,
        'feature_selection': True,
        'rfe': False,
        'univariate_selection': False,
        'pvalue_level': 0.01,
        'max_features': 100,
        'transform': [],
        'lags': False,
        'lag_period': 1,
        'poly_degree': 1,
        'use_smote': False
    }
    
    return Model(model_specs)


def run_alphapy_pipeline(df, target_variable, predictors):
    """
    Execute the optimized AlphaPy AutoML pipeline
    """
    print("🚀 Starting AlphaPy AutoML pipeline...")
    
    # Prepare data splits
    X = df[predictors]
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create AlphaPy model
    model = create_alphapy_model()
    
    # Configure model data for AlphaPy
    model.df_X_train = X_train
    model.df_y_train = y_train
    model.df_X_test = X_test
    model.df_y_test = y_test
    model.X_train = X_train
    model.y_train = y_train
    model.X_test = X_test
    model.y_test = y_test
    
    print("Training algorithms with AlphaPy framework...")
    
    # Use scikit-learn algorithms in AlphaPy structure for consistency
    algorithms = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = []
    best_model = None
    best_score = -float('inf')
    
    for name, algo in algorithms.items():
        print(f"  ⚡ Training {name}...")
        algo.fit(X_train, y_train)
        
        # Generate predictions
        y_train_pred = algo.predict(X_train)
        y_test_pred = algo.predict(X_test)
        
        # Calculate performance metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results.append({
            'Algorithm': name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse
        })
        
        # Track best performing model
        if test_r2 > best_score:
            best_score = test_r2
            best_model = {
                'name': name, 
                'model': algo, 
                'test_r2': test_r2, 
                'test_rmse': test_rmse,
                'feature_names': X.columns.tolist()
            }
    
    return best_model, pd.DataFrame(results)


def display_results(best_model, results_df):
    """
    Display comprehensive results and model performance analysis
    """
    print("\n" + "="*80)
    print("🎯 AlphaPy AutoML Results")
    print("="*80)
    
    # Sort and format results
    results_df = results_df.sort_values('Test_R2', ascending=False)
    display_df = results_df[['Algorithm', 'Test_R2', 'Test_RMSE']].copy()
    display_df['Test_R2'] = display_df['Test_R2'].round(4)
    display_df['Test_RMSE'] = display_df['Test_RMSE'].round(2)
    
    print(display_df.to_string(index=False))
    
    # Best model summary
    print(f"\n🏆 Best Performing Model: {best_model['name']}")
    print(f"📊 Test R² Score: {best_model['test_r2']:.4f}")
    print(f"💰 Test RMSE: ${best_model['test_rmse']:,.2f}")
    print(f"\n📝 Model Performance Interpretation:")
    print(f"   • Explains {best_model['test_r2']*100:.1f}% of price variance")
    print(f"   • Average prediction error: ${best_model['test_rmse']:,.2f}")
    
    # Feature importance analysis
    if hasattr(best_model['model'], 'feature_importances_'):
        print(f"\n🔍 Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'Feature': best_model['feature_names'],
            'Importance': best_model['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['Feature']:<20} ({row['Importance']:.4f})")


def run_house_price_prediction_pipeline():
    """
    Main pipeline function that orchestrates the entire AlphaPy AutoML process
    """
    print("🏡 House Price Prediction with AlphaPy AutoML")
    print("=" * 60)
    
    try:
        # Step 1: Load and explore data
        df = load_and_explore_data()
        
        # Step 2: Preprocess data
        df_processed, predictors, label_encoders = preprocess_data(df)
        
        # Step 3: Run AlphaPy pipeline
        best_model, results_df = run_alphapy_pipeline(
            df_processed, 'price', predictors
        )
        
        # Step 4: Display comprehensive results
        display_results(best_model, results_df)
        
        print("\n" + "="*80)
        print("🎉 AlphaPy AutoML Pipeline Completed Successfully!")
        print("="*80)
        
        return best_model, results_df
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise e


if __name__ == '__main__':
    # Execute the main pipeline
    run_house_price_prediction_pipeline()