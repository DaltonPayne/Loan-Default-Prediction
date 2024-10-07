
# Loan Default Prediction System

## Overview

This project is designed to predict loan defaults using machine learning techniques, specifically XGBoost. The project utilizes a dataset of loan applications and analyzes various borrower and loan attributes to determine the likelihood of defaulting on a loan.

This system is structured into three main components:
1. **Exploratory Data Analysis (EDA)**: A notebook providing a detailed analysis and visualization of the dataset.
2. **Model Training**: A notebook for training and evaluating an XGBoost model for predicting loan defaults, including hyperparameter tuning and performance evaluation.
3. **Inference Script**: A Python script to input loan features and predict the probability of default in real time.

## Dataset

The dataset used in this project is `Loan_default.csv` provided by Coursera. It contains borrower and loan-related features, including:
- Borrower demographics (Age, Income, Employment, etc.)
- Loan details (Loan Amount, Interest Rate, Loan Term, etc.)
- Other relevant variables (Education, Loan Purpose, Marital Status, etc.)


## Prerequisites

To run this project locally, you'll need the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `pickle`

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Exploratory Data Analysis (EDA)

Run the `loan_default_analysis.ipynb` notebook to analyze the dataset. This notebook:
- Cleans the data and handles missing values.
- Provides visualizations for understanding feature distributions and relationships.
- Computes summary statistics and insights into key features affecting loan default.

### 2. Model Training

The `loan_default_prediction.ipynb` notebook:
- Preprocesses the data, including handling categorical variables and class imbalance.
- Trains an XGBoost model, utilizing RandomizedSearchCV for hyperparameter tuning.
- Evaluates the model using ROC AUC score, log loss, and other performance metrics.
- Saves the trained model (`xgboost_loan_default_model.pkl`) and label encoders (`label_encoders.pkl`) for future inference.

### 3. Inference

The `inference.py` script allows you to input custom loan data and predict the probability of default.

To run the inference script, execute:

```bash
python scripts/inference.py
```

When prompted, enter the required loan data such as age, income, loan amount, etc. The script will then output the predicted probability of default based on the model.

### Example

```bash
Enter age: 35
Enter income: 50000
Enter loan amount: 15000
Enter interest rate: 5.5
Enter education level (High School, Bachelor's, Master's): Bachelor's 
Enter loan term (in months): 60
Enter months employed: 120
Enter employment type (Full-time, Part-time, Unemployed): Full-time
Enter marital status (Single, Married, Divorced): Married
Has a cosigner? (Yes/No): no
Has dependents? (Yes/No): Yes
Enter loan purpose (Auto, Business, Other): Auto
The predicted probability of defaulting on the loan is: 2.67%
```

## Model Evaluation

The model is evaluated using various metrics, including:
- **ROC AUC Score**: Measures the ability of the model to distinguish between defaulters and non-defaulters.
- **Log Loss**: Evaluates the accuracy of probabilistic predictions.
- **Classification Report**: Includes precision, recall, F1-score, and support.
- **Confusion Matrix**: Summarizes correct and incorrect predictions.

### Feature Importance

The feature importance plot, generated in the model training notebook, helps in understanding which features contribute most to the model's predictions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
