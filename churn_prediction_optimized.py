# Churn Prediction Model with Optimized Data Processing
# Author: Hose I. Rad
# Date: Oct. 19th, 2024

"""
This script builds a churn prediction model for an eCommerce platform using user event data.
Given the large dataset size, the code is optimized for efficient data processing.

Key Features:
- Processes data in manageable chunks to conserve memory.
- Utilizes multiprocessing for faster file processing.
- Extracts and engineers features relevant to churn prediction.
- Addresses class imbalance using SMOTE.
- Performs hyperparameter tuning for optimal model performance.
- Trains an XGBoost classifier to predict user churn.
"""

# ------------------------------
# Import Necessary Libraries
# ------------------------------
import sys
import subprocess
import logging
import gc

def install_library(package):
    """
    Installs a Python library using pip if it's not already installed.

    Args:
        package (str): The name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Define required libraries with correct package names
required_libraries = {
    'pandas': 'pd',
    'numpy': 'np',
    'matplotlib': 'plt',
    'seaborn': 'sns',
    'scikit-learn': None,
    'xgboost': None,
    'imbalanced-learn': 'imblearn',
    'joblib': None
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("churn_prediction.log"),
        logging.StreamHandler()
    ]
)

# Install any missing libraries
logging.info("Checking and installing missing libraries...")
for lib, alias in required_libraries.items():
    try:
        if alias:
            globals()[alias] = __import__(lib)
        else:
            __import__(lib)
    except ImportError:
        logging.info(f"Installing {lib}...")
        install_library(lib)
        # Re-import after installation
        if alias:
            globals()[alias] = __import__(lib)
        else:
            __import__(lib)
logging.info("All libraries are ready.")

# Now import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
from multiprocessing import Pool, cpu_count
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seaborn style for plots
sns.set(style='whitegrid')

# ------------------------------
# Data Processing Functions
# ------------------------------

def process_file(file_path):
    """
    Processes a single CSV file to extract relevant features for churn prediction.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Extracted features from the file.
    """
    logging.info(f"Processing file: {file_path}")
    compression = 'gzip' if file_path.endswith('.gz') else None
    feature_chunks = []

    try:
        # Read the file in chunks to handle large data
        for chunk in pd.read_csv(
            file_path,
            compression=compression,
            parse_dates=['event_time'],
            low_memory=False,
            chunksize=5_000_000,  # Adjust based on system memory
            dtype={
                'event_type': 'category',
                'product_id': 'int32',
                'category_id': 'float32',
                'category_code': 'category',
                'brand': 'category',
                'price': 'float32',
                'user_id': 'int32',
                'user_session': 'object'
            }
        ):
            # Fill missing values
            chunk['category_code'].fillna('unknown', inplace=True)
            chunk['brand'].fillna('unknown', inplace=True)

            # Extract time-based features
            chunk['event_time'] = chunk['event_time'].dt.tz_localize(None)
            chunk['event_date'] = chunk['event_time'].dt.date
            chunk['month'] = chunk['event_time'].dt.to_period('M')

            # Aggregate features by user and month
            features = chunk.groupby(['user_id', 'month']).agg(
                total_events=('event_type', 'count'),
                unique_event_types=('event_type', 'nunique'),
                num_views=('event_type', lambda x: (x == 'view').sum()),
                num_carts=('event_type', lambda x: (x == 'cart').sum()),
                num_purchases=('event_type', lambda x: (x == 'purchase').sum()),
                num_remove_from_cart=('event_type', lambda x: (x == 'remove_from_cart').sum()),
                num_unique_products=('product_id', 'nunique'),
                num_unique_categories=('category_code', 'nunique'),
                avg_price=('price', 'mean'),
                max_price=('price', 'max'),
                min_price=('price', 'min'),
                num_sessions=('user_session', 'nunique'),
                active_days=('event_date', 'nunique'),
                first_event_time=('event_time', 'min'),
                last_event_time=('event_time', 'max')
            ).reset_index()

            feature_chunks.append(features)

        if feature_chunks:
            # Combine all chunks
            df_features = pd.concat(feature_chunks, ignore_index=True)
            del feature_chunks  # Free memory
            gc.collect()

            # Further aggregate in case of overlapping data
            df_features = df_features.groupby(['user_id', 'month']).agg({
                'total_events': 'sum',
                'unique_event_types': 'sum',
                'num_views': 'sum',
                'num_carts': 'sum',
                'num_purchases': 'sum',
                'num_remove_from_cart': 'sum',
                'num_unique_products': 'sum',
                'num_unique_categories': 'sum',
                'avg_price': 'mean',
                'max_price': 'max',
                'min_price': 'min',
                'num_sessions': 'sum',
                'active_days': 'sum',
                'first_event_time': 'min',
                'last_event_time': 'max'
            }).reset_index()

            # Feature engineering
            df_features['view_to_purchase_ratio'] = df_features['num_views'] / df_features['num_purchases']
            df_features['cart_to_purchase_ratio'] = df_features['num_carts'] / df_features['num_purchases']
            df_features.replace([np.inf, np.nan], 0, inplace=True)

            df_features['session_duration'] = (df_features['last_event_time'] - df_features['first_event_time']).dt.total_seconds()
            df_features['session_duration'].fillna(0, inplace=True)

            max_date = df_features['last_event_time'].max()
            df_features['recency'] = (max_date - df_features['last_event_time']).dt.days

            # Remove unnecessary columns
            df_features.drop(['first_event_time', 'last_event_time'], axis=1, inplace=True)

            return df_features
        else:
            logging.warning(f"No data processed for file: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

def process_data(data_path):
    """
    Processes all CSV files in the specified directory to extract features.

    Args:
        data_path (str): Directory containing CSV data files.

    Returns:
        pd.DataFrame: Combined features from all files.
    """
    if not os.path.exists(data_path):
        logging.error(f"Data directory '{data_path}' does not exist.")
        sys.exit(1)

    files = sorted(glob(os.path.join(data_path, '*.csv*')))
    if not files:
        logging.error(f"No CSV files found in '{data_path}'.")
        sys.exit(1)

    logging.info(f"Found {len(files)} files to process.")

    with Pool(cpu_count()) as pool:
        features_list = pool.map(process_file, files)

    # Remove empty DataFrames
    features_list = [df for df in features_list if not df.empty]

    if features_list:
        features_df = pd.concat(features_list, ignore_index=True)
        features_df.to_pickle('features_df.pkl')  # Save for future use
        logging.info("Features saved to 'features_df.pkl'.")
        return features_df
    else:
        logging.warning("No data was processed.")
        return pd.DataFrame()

def define_churn(features_df):
    """
    Defines churn labels based on user activity in subsequent months.

    Args:
        features_df (pd.DataFrame): DataFrame with user features.

    Returns:
        pd.DataFrame: DataFrame with churn labels.
    """
    months = sorted(features_df['month'].unique())
    churn_records = []

    for idx, month in enumerate(months[:-2]):
        current_month = month
        next_two_months = months[idx + 1: idx + 3]

        users_current = set(features_df[features_df['month'] == current_month]['user_id'])
        users_next = set(features_df[features_df['month'].isin(next_two_months)]['user_id'])

        for user in users_current:
            churn = 1 if user not in users_next else 0
            churn_records.append({'user_id': user, 'month': current_month, 'churn': churn})

    churn_df = pd.DataFrame(churn_records)
    return churn_df

def prepare_data(features_df, churn_df):
    """
    Merges features with churn labels and handles missing values.

    Args:
        features_df (pd.DataFrame): DataFrame with user features.
        churn_df (pd.DataFrame): DataFrame with churn labels.

    Returns:
        pd.DataFrame: Merged and cleaned dataset ready for modeling.
    """
    # Merge on user_id and month
    data = pd.merge(features_df, churn_df, on=['user_id', 'month'], how='inner')

    # Exclude 'month' from filling missing values
    non_period_cols = [col for col in data.columns if col != 'month']

    # Fill missing values in non-period columns
    data[non_period_cols] = data[non_period_cols].fillna(0)

    return data

# ------------------------------
# Model Training Function
# ------------------------------

def train_model(X_train, y_train):
    """
    Trains an XGBoost classifier with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        Pipeline: Trained machine learning pipeline.
    """
    # Address class imbalance
    smote = SMOTE(random_state=42)

    # Initialize XGBoost classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', smote),
        ('classifier', xgb)
    ])

    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
    }

    # Setup randomized search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Train the model
    search.fit(X_train, y_train)
    logging.info(f"Optimal parameters: {search.best_params_}")

    return search.best_estimator_

# ------------------------------
# Main Execution Block
# ------------------------------

def main():
    """
    Main function to execute the churn prediction workflow.
    """
    logging.info("Starting the churn prediction workflow...")

    # Define data directory
    data_path = '/hose/Decs/Zeal/churn_predict'  # Update this path as needed
    logging.info(f"Data directory set to: {data_path}")

    # Load or process features
    features_file = 'features_df.pkl'
    if os.path.exists(features_file):
        logging.info(f"Loading existing features from '{features_file}'...")
        features_df = pd.read_pickle(features_file)
    else:
        logging.info("Processing data files to extract features...")
        features_df = process_data(data_path)

    # Display a snapshot of the features
    if features_df.empty:
        logging.warning("No features found. Exiting the workflow.")
        sys.exit(1)
    else:
        logging.info("Sample of Extracted Features:")
        logging.info(f"\n{features_df.head()}")

    # Load or define churn labels
    churn_file = 'churn_df.pkl'
    if os.path.exists(churn_file):
        logging.info(f"Loading existing churn labels from '{churn_file}'...")
        churn_df = pd.read_pickle(churn_file)
    else:
        logging.info("Defining churn labels based on user activity...")
        churn_df = define_churn(features_df)
        churn_df.to_pickle(churn_file)
        logging.info(f"Churn labels saved to '{churn_file}'.")

    # Merge features with churn labels
    logging.info("Merging features with churn labels...")
    data = prepare_data(features_df, churn_df)
    logging.info("Sample of Merged Data:")
    logging.info(f"\n{data.head()}")

    # Save the prepared dataset
    prepared_data_file = 'prepared_data.pkl'
    data.to_pickle(prepared_data_file)
    logging.info(f"Prepared dataset saved to '{prepared_data_file}'.")

    # Split data into training and testing sets based on months
    logging.info("Splitting data into training and testing sets...")
    months = sorted(data['month'].unique())
    if len(months) < 3:
        logging.error("Not enough months of data to split into training and testing sets.")
        sys.exit(1)
    train_months = months[:-2]  # All months except the last two
    test_months = months[-2:]    # Last two months for testing

    train_data = data[data['month'].isin(train_months)].reset_index(drop=True)
    test_data = data[data['month'].isin(test_months)].reset_index(drop=True)

    # Separate features and target variable
    logging.info("Separating features and target variable...")
    X_train = train_data.drop(['user_id', 'month', 'churn'], axis=1)
    y_train = train_data['churn']
    X_test = test_data.drop(['user_id', 'month', 'churn'], axis=1)
    y_test = test_data['churn']

    # Train or load the model
    model_file = 'churn_prediction_model.pkl'
    if os.path.exists(model_file):
        logging.info(f"Loading existing model from '{model_file}'...")
        model = joblib.load(model_file)
    else:
        logging.info("Training the churn prediction model...")
        model = train_model(X_train, y_train)
        joblib.dump(model, model_file)
        logging.info(f"Model trained and saved to '{model_file}'.")

    # Make predictions on the test set
    logging.info("Generating predictions on the test set...")
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    logging.info("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])
    logging.info(f"\n{report}")

    # Calculate ROC-AUC Score
    if hasattr(model.named_steps['classifier'], 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        logging.info(f"ROC-AUC Score: {roc_auc:.4f}")
    else:
        logging.warning("ROC-AUC Score cannot be calculated.")

    # Plot the confusion matrix
    logging.info("Displaying the confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Retained', 'Churned'],
        yticklabels=['Retained', 'Churned']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    logging.info("Confusion matrix saved as 'confusion_matrix.png'.")

    # Show feature importance if available
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        logging.info("Plotting feature importance...")
        importances = model.named_steps['classifier'].feature_importances_
        feat_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

        # Plot top 10 features
        plt.figure(figsize=(8, 6))
        feat_importances.head(10).plot(kind='barh')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title('Top 10 Important Features')
        plt.gca().invert_yaxis()
        plt.savefig('feature_importance.png')
        plt.close()
        logging.info("Feature importance saved as 'feature_importance.png'.")
    else:
        logging.warning("Feature importance is not available.")

if __name__ == '__main__':
    main()
