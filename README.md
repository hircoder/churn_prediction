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

## Features

- **Efficient Data Processing:** Utilizes **Dask** for handling large datasets by processing data in manageable chunks.
- **Multiprocessing:** Employs Python's `multiprocessing` library to accelerate file processing.
- **Feature Engineering:** Extracts and engineers relevant features essential for accurate churn prediction.
- **Class Imbalance Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance in the dataset.
- **Hyperparameter Tuning:** Uses **RandomizedSearchCV** for optimizing model parameters to enhance performance.
- **Model Training:** Trains an **XGBoost** classifier known for its robustness and accuracy in classification tasks.
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
imbalanced-learn
joblib


## Usage
Prepare  Data

Ensure that your user event data CSV.gz (or .CSV) files are placed in the designated data directory. Update the data_path variable in the script to point to your data location.

data_path = '/path/to/your/data_directory'  

Execute the churn prediction script:
python churn_prediction_optimized.py

The script will display logs indicating the progress of data processing, feature extraction, model training, and evaluation.

## View Results

After successful execution, the script will output:

Classification Report: Detailed metrics on model performance.
ROC-AUC Score: Evaluation metric to assess the model's ability to distinguish between classes.
Confusion Matrix: Visual representation of prediction outcomes.
Feature Importance: Insights into which features significantly impact churn predictions.

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
