# Housing Price Prediction - Assignment
# Hasnat bin sayed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# COMPETITION METRIC FUNCTION
# =============================================================================

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    """
    Compute RMSLE (Root Mean Squared Logarithmic Error) - competition metric
    """
    # Ensure no negative values for log calculation
    y_test = np.maximum(y_test, 0)
    y_pred = np.maximum(y_pred, 0)
    
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

# =============================================================================
# 1. DATA SETUP AND LOADING
# =============================================================================

def validate_dataset(df):
    """Validate dataset meets competition requirements"""
    print(" Validating dataset...")
    assert 'SalePrice' in df.columns, "Target variable 'SalePrice' not found"
    assert len(df) > 1000, "Dataset too small"
    assert df.shape[1] > 10, "Insufficient features in dataset"
    print("✓ Dataset validation passed")

def load_data(file_path):
    """
    Load housing dataset from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully from {file_path}")
        print(f"✓ Dataset shape: {df.shape}")
        validate_dataset(df)
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        print("Please ensure train.csv is in the data/ folder")
        return None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

# =============================================================================
# 2. FEATURE SELECTION
# =============================================================================

def select_features(df):
    """
    Select minimum 2 continuous and 2 categorical features as per instructions
    """
    print("\n" + "="*50)
    print("FEATURE SELECTION")
    print("="*50)
    
    # Display available columns
    print("Available columns in dataset:")
    print(f"Total columns: {len(df.columns)}")
    
    # Separate continuous and categorical features
    continuous_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from continuous features
    if 'SalePrice' in continuous_features:
        continuous_features.remove('SalePrice')
    if 'Id' in continuous_features:
        continuous_features.remove('Id')
    
    print(f"\nContinuous features ({len(continuous_features)}): {continuous_features[:8]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features[:8]}...")
    
    # Select specific features with clear rationale
    selected_continuous = []
    selected_categorical = []
    
    # Select meaningful continuous features
    preferred_continuous = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'GarageArea']
    for feature in preferred_continuous:
        if feature in continuous_features and len(selected_continuous) < 2:
            selected_continuous.append(feature)
    
    # Select meaningful categorical features
    preferred_categorical = ['Neighborhood', 'HouseStyle', 'BldgType', 'RoofStyle']
    for feature in preferred_categorical:
        if feature in categorical_features and len(selected_categorical) < 2:
            selected_categorical.append(feature)
    
    # Fallback selection if preferred features not available
    if len(selected_continuous) < 2:
        additional_continuous = [f for f in continuous_features if f not in selected_continuous]
        selected_continuous.extend(additional_continuous[:2-len(selected_continuous)])
    
    if len(selected_categorical) < 2:
        additional_categorical = [f for f in categorical_features if f not in selected_categorical]
        selected_categorical.extend(additional_categorical[:2-len(selected_categorical)])
    
    print(f"\n SELECTED CONTINUOUS FEATURES ({len(selected_continuous)}):")
    for feature in selected_continuous:
        print(f"   - {feature}")
    
    print(f"\n SELECTED CATEGORICAL FEATURES ({len(selected_categorical)}):")
    for feature in selected_categorical:
        print(f"   - {feature}")
    
    # Feature selection rationale
    print(f"\n FEATURE SELECTION RATIONALE:")
    print(f"   Continuous: {selected_continuous[0]} (living area), {selected_continuous[1]} (basement size) - direct size indicators")
    print(f"   Categorical: {selected_categorical[0]} (location), {selected_categorical[1]} (architecture style) - important categorical factors")
    
    return selected_continuous, selected_categorical

# =============================================================================
# 3. FEATURE PROCESSING
# =============================================================================

def preprocess_features(df, continuous_features, categorical_features):
    """
    Process, scale and encode features as per instructions
    WITHOUT using ColumnTransformer or Pipeline
    """
    print("\n" + "="*50)
    print("FEATURE PROCESSING")
    print("="*50)
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Handle missing values
    print("\n Handling missing values...")
    
    # Continuous features: fill with median
    for feature in continuous_features:
        if df_processed[feature].isnull().sum() > 0:
            median_val = df_processed[feature].median()
            df_processed[feature].fillna(median_val, inplace=True)
            print(f"   - {feature}: filled {df_processed[feature].isnull().sum()} missing values with median {median_val:.2f}")
        else:
            print(f"   - {feature}: no missing values")
    
    # Categorical features: fill with mode
    for feature in categorical_features:
        if df_processed[feature].isnull().sum() > 0:
            mode_val = df_processed[feature].mode()[0] if not df_processed[feature].mode().empty else 'Unknown'
            df_processed[feature].fillna(mode_val, inplace=True)
            print(f"   - {feature}: filled {df_processed[feature].isnull().sum()} missing values with mode '{mode_val}'")
        else:
            print(f"   - {feature}: no missing values")
    
    # Scale continuous features
    print("\n Scaling continuous features...")
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(df_processed[continuous_features])
    
    # Convert back to DataFrame with clear naming
    scaled_df = pd.DataFrame(scaled_continuous, columns=[f"{feat}_scaled" for feat in continuous_features])
    print(f"   - Scaled {len(continuous_features)} continuous features using StandardScaler")
    
    # Encode categorical features
    print("\n Encoding categorical features...")
    label_encoders = {}
    encoded_features = []
    
    for feature in categorical_features:
        le = LabelEncoder()
        encoded_values = le.fit_transform(df_processed[feature].astype(str))
        encoded_df = pd.DataFrame(encoded_values, columns=[f"{feature}_encoded"])
        encoded_features.append(encoded_df)
        label_encoders[feature] = le
        print(f"   - {feature}: encoded {len(le.classes_)} categories using LabelEncoder")
    
    # Combine all processed features
    final_features_df = pd.concat([scaled_df] + encoded_features, axis=1)
    
    print(f"\n Final processed features shape: {final_features_df.shape}")
    print(f" Processed feature names: {list(final_features_df.columns)}")
    
    return final_features_df, scaler, label_encoders

# =============================================================================
# 4. MODEL TRAINING
# =============================================================================

def train_models(X_train, y_train):
    """
    Train multiple regression models
    """
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f" {name} training completed")
        print(f"   - Training samples: {X_train.shape[0]}")
        print(f"   - Features: {X_train.shape[1]}")
    
    return trained_models

# =============================================================================
# 5. MODEL EVALUATION
# =============================================================================

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate models using competition metric RMSLE
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    results = {}
    
    for name, model in trained_models.items():
        print(f"\n Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Ensure no negative predictions for log calculation
        y_pred = np.maximum(y_pred, 0)
        
        # Calculate RMSLE - competition metric
        rmsle = compute_rmsle(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'rmsle': rmsle,
            'predictions': y_pred
        }
        
        print(f" {name} RMSLE: {rmsle}")
        print(f"   - Test samples: {len(y_test)}")
        print(f"   - Min prediction: ${y_pred.min():,.2f}")
        print(f"   - Max prediction: ${y_pred.max():,.2f}")
        print(f"   - Mean prediction: ${y_pred.mean():,.2f}")
    
    return results

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_feature_distributions(df, continuous_features, categorical_features):
    """
    Plot distributions of selected features
    """
    print("\n Plotting feature distributions...")
    
    # Plot continuous features
    if continuous_features:
        fig, axes = plt.subplots(1, len(continuous_features), figsize=(15, 5))
        if len(continuous_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(continuous_features):
            axes[i].hist(df[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    # Plot categorical features
    if categorical_features:
        fig, axes = plt.subplots(1, len(categorical_features), figsize=(15, 5))
        if len(categorical_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(categorical_features):
            value_counts = df[feature].value_counts().head(8)  # Top 8 categories
            axes[i].bar(value_counts.index, value_counts.values, color='lightcoral', edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def plot_target_distribution(df):
    """
    Plot distribution of target variable
    """
    print("\n Plotting target variable distribution...")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['SalePrice'], bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribution of SalePrice', fontsize=12, fontweight='bold')
    plt.xlabel('Sale Price ($)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df['SalePrice']), bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Log(SalePrice)', fontsize=12, fontweight='bold')
    plt.xlabel('Log(Sale Price)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main function executing the complete modeling pipeline
    """
    print("="*70)
    print("HOUSE PRICES PREDICTION - PW1 ASSIGNMENT")
    print("="*70)
    
    # =========================================================================
    # 1. DATA SETUP AND LOADING
    # =========================================================================
    print("\n STEP 1: DATA SETUP AND LOADING")
    print("-" * 40)
    
    # Try multiple possible file locations
    possible_paths = [
        './data/train.csv',      # Data folder in current directory
        '../data/train.csv',     # Data folder in parent directory  
        './train.csv',           # Current directory
        '../train.csv',          # Parent directory
    ]
    
    df = None
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"✓ Found dataset at: {file_path}")
            df = load_data(file_path)
            if df is not None:
                break
    
    if df is None:
        print(" Failed to load data. Please ensure train.csv is in one of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        
        # Show available files for debugging
        print("\n Available files in current directory:")
        try:
            files = os.listdir('.')
            csv_files = [f for f in files if f.endswith('.csv')]
            folders = [f for f in files if os.path.isdir(f)]
            
            if csv_files:
                print("CSV files:", csv_files)
            if folders:
                print("Folders:", folders)
        except Exception as e:
            print(f"Error listing files: {e}")
        return
    
    # Display basic dataset information
    print(f"\n DATASET OVERVIEW:")
    print(f"   - Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display first few rows (limited to 5 as per instructions)
    print(f"\n FIRST 5 ROWS OF THE DATASET:")
    print(df.head().to_string())
    
    # Display target variable information
    if 'SalePrice' in df.columns:
        print(f"\n TARGET VARIABLE 'SalePrice' ANALYSIS:")
        print(f"   - Min: ${df['SalePrice'].min():,.2f}")
        print(f"   - Max: ${df['SalePrice'].max():,.2f}")
        print(f"   - Mean: ${df['SalePrice'].mean():,.2f}")
        print(f"   - Median: ${df['SalePrice'].median():,.2f}")
        print(f"   - Standard Deviation: ${df['SalePrice'].std():,.2f}")
    
    # Plot target distribution
    plot_target_distribution(df)
    
    # =========================================================================
    # 2. FEATURE SELECTION
    # =========================================================================
    print("\n\n STEP 2: FEATURE SELECTION")
    print("-" * 40)
    
    continuous_features, categorical_features = select_features(df)
    
    # Display selected features statistics
    print(f"\n SELECTED CONTINUOUS FEATURES STATISTICS:")
    print(df[continuous_features].describe().round(2))
    
    print(f"\n SELECTED CATEGORICAL FEATURES VALUE COUNTS:")
    for feature in categorical_features:
        print(f"\n{feature}:")
        print(df[feature].value_counts().head(5))  # Show top 5 only
    
    # Plot feature distributions
    plot_feature_distributions(df, continuous_features, categorical_features)
    
    # =========================================================================
    # 3. FEATURE PROCESSING
    # =========================================================================
    print("\n\n STEP 3: FEATURE PROCESSING")
    print("-" * 40)
    
    # Prepare features and target
    X_raw = df[continuous_features + categorical_features]
    y = df['SalePrice']
    
    print(f" Raw features shape: {X_raw.shape}")
    print(f" Target shape: {y.shape}")
    
    # Process features (scale continuous, encode categorical)
    X_processed, scaler, label_encoders = preprocess_features(
        df, continuous_features, categorical_features
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\n DATA SPLIT RESULTS:")
    print(f"   - Training set: {X_train.shape} (80% of data)")
    print(f"   - Testing set: {X_test.shape} (20% of data)")
    print(f"   - Training target: {y_train.shape}")
    print(f"   - Testing target: {y_test.shape}")
    print(f"   - Split configuration: test_size=0.2, random_state=42, shuffle=True")
    
    # =========================================================================
    # 4. MODEL TRAINING
    # =========================================================================
    print("\n\n STEP 4: MODEL TRAINING")
    print("-" * 40)
    
    trained_models = train_models(X_train, y_train)
    
    # =========================================================================
    # 5. MODEL EVALUATION
    # =========================================================================
    print("\n\n STEP 5: MODEL EVALUATION")
    print("-" * 40)
    
    results = evaluate_models(trained_models, X_test, y_test)
    
    # =========================================================================
    # FINAL RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmsle'])
    best_rmsle = results[best_model_name]['rmsle']
    
    print(f"\n BEST PERFORMING MODEL: {best_model_name}")
    print(f" BEST RMSLE SCORE: {best_rmsle}")
    
    print(f"\n ALL MODEL RESULTS (sorted by RMSLE):")
    models_sorted = sorted(results.items(), key=lambda x: x[1]['rmsle'])
    for i, (name, result) in enumerate(models_sorted, 1):
        print(f"   {i}. {name}: RMSLE = {result['rmsle']}")
    
  
    
    # Show feature importance if available
    if best_model_name == 'Random Forest':
        print(f"\n RANDOM FOREST FEATURE IMPORTANCE (Top 10):")
        best_model = results[best_model_name]['model']
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).round(4))
    


# Execute the main pipeline
if __name__ == "__main__":
    main()