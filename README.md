# Churn Prediction Model with Optimized Data Processing

**Author:** Hose I. Rad  
**Date:** October 19th, 2024

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)


## Introduction

This project entails building a **Churn Prediction Model** for an eCommerce platform using extensive user event data. Given the substantial size of the dataset, the model is optimized to handle large-scale data processing efficiently within a limited memory environment.

Churn prediction is important for businesses to identify and retain customers who are likely to discontinue using their services. By leveraging advanced data processing techniques and machine learning algorithms, this model aims to provide accurate churn predictions, enabling proactive customer retention strategies.

The project includes two key notebooks:

churn_predict_optimized_org.ipynb: This notebook is responsible for processing the raw user event data, performing feature engineering, and generating a cleaned dataset. It saves this dataset as prepared_data.pkl for future use, which optimizes subsequent model training steps.
churn_LGBM_optimized.ipynb: This notebook focuses exclusively on training the churn prediction model using the LightGBM algorithm. It loads the intermediate data (prepared_data.pkl) to skip the computationally expensive data preparation step, allowing the model training to proceed directly.

### Data and Model Download
You can download the preprocessed data (`prepared_data.pkl`) and the trained model (`churn_prediction_model.pkl`) from the following link:

[Download prepared data and model](https://drive.google.com/drive/folders/1W9SVvLfelyBQFw1uOFa-Js0HJCnCHx4A?usp=sharing)

This allows you to either directly use the files to retrain the model or make improvements on the existing model.


## Features

- **Efficient Data Processing:** Tried to Optimize data handling large datasets by processing data in manageable chunks.
- **Multiprocessing:** Employs Python's `multiprocessing` library to accelerate file processing.
- **Feature Engineering:** Extracts and engineers relevant features essential for accurate churn prediction.
- **Class Imbalance Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance in the dataset.
- **Hyperparameter Tuning:** Uses **RandomizedSearchCV** for optimizing model parameters to enhance performance.
- **Model Training:** churn_predict_optimized.ipynb Trains an **XGBoost** classifier known for its robustness and accuracy in classification tasks.
      - LightGBM model in churn_LGBM_optimized.ipynb, which directly uses the prepared data to train and fine-tune the model efficiently.
- **Model Persistence:** Saves intermediate results and the final trained model for future use, eliminating the need for reprocessing large data files.

## Prerequisites

Ensure that you have the following installed on your system:

- **Python 3.7 or higher**
- **pip** (Python package installer)
- **Virtual Environment** (recommended)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/hircoder/churn_prediction.git
   cd churn-prediction
Create a Virtual Environment

It's advisable to use a virtual environment to manage dependencies.

python3 -m venv venv
source venv/bin/activate  


Install Required Libraries
The script includes a mechanism to install any missing libraries automatically. However, you can manually install them using:
pip install -r requirements.txt

If requirements.txt is not provided, ensure the following libraries are installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imbalanced-learn
joblib


## Usage
Prepare  Data

Prepare Data (churn_predict_optimized.ipynb)
1. Ensure that your user event data files (in CSV or CSV.gz format) are placed in the designated data directory.
2. Open and run the churn_predict_optimized_org.ipynb notebook.
3.This notebook will process the raw data, perform feature engineering, and save the processed data as prepared_data.pkl.

Train Model (churn_LGBM_optimized.ipynb)
1. Once the prepared_data.pkl file is generated, open and run the churn_LGBM_optimized.ipynb notebook.
2. This notebook loads the preprocessed data, applies the LightGBM algorithm, and performs hyperparameter tuning.
3. The final trained model will be saved as churn_prediction_model.pkl.

## View Results

After running churn_predict_optimized.ipynb:
   - You will obtain the intermediate prepared_data.pkl file, which contains all the engineered features required for model training.

After running churn_LGBM_optimized.ipynb:
The notebook will output the following results:
   - Classification Report: Detailed metrics on model performance, such as precision, recall, and F1-score.
   - ROC-AUC Score: A performance metric indicating how well the model distinguishes between churned and retained customers.
   - Confusion Matrix: A visual representation of prediction accuracy, showing true positive/negative and false positive/negative predictions.
   - Feature Importance: Insights into which features contribute the most to the prediction of customer churn.

## Project Structure

churn-prediction/
│
├── churn_prediction_optimized.py        # Main script for data processing and model training
├── features_df.pkl            # Pickle file containing extracted features
├── churn_df.pkl               # Pickle file containing churn labels
├── prepared_data.pkl          # Pickle file with merged features and labels
├── churn_prediction_model.pkl # Saved trained model
├── README.md                  # Project documentation
└── Approach_and_Challenges.md # Documentation of approach and challenges

## Results
Upon successful execution, the model provides classification insights into customer churn. The evaluation metrics indicate the model's effectiveness in predicting churn, while feature importance highlights key factors influencing customer behavior.

## Contributing
Contributions are welcome! If you have suggestions for improvements or encounter any issues, feel free to open an issue or submit a pull request.

Fork the repository.
Create a new branch: git checkout -b feature-name.
Commit your changes: git commit -m 'Add some feature'.
Push to the branch: git push origin feature-name.
Open a pull request.
