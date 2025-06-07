
# SUV Purchase Prediction using Linear Regression

This project demonstrates a machine learning workflow to predict whether a customer will purchase an SUV based on demographic and salary information. While the primary model used is Linear Regression for demonstration, the problem is inherently a binary classification task, for which Logistic Regression is typically more suitable.

## Table of Contents
1.  [Introduction & Theory](#introduction--theory)
2.  [Project Structure](#project-structure)
3.  [Setup](#setup)
4.  [Code Steps](#code-steps)
    * [Step 1: Data Collection](#step-1-data-collection)
    * [Step 2: Data Analyzing](#step-2-data-analyzing)
    * [Step 3: Data Wrangling](#step-3-data-wrangling)
    * [Step 4: Training and Testing](#step-4-training-and-testing)
    * [Step 5: Accuracy Check and Evaluation](#step-5-accuracy-check-and-evaluation)
5.  [Conclusion](#conclusion)

## Introduction & Theory

### What is this project about?
The goal of this project is to build a predictive model that can determine, based on certain user attributes like age and estimated salary, whether they are likely to purchase an SUV. This is a common task in marketing and sales to identify potential customers.

### Independent and Dependent Variables
In machine learning, we categorize variables based on their role in prediction:
* **Dependent Variable (Target Variable / `y`):** This is the variable we want to predict. In this project, it's the `Purchased` column (0 for No, 1 for Yes).
* **Independent Variables (Features / Predictors / `X`):** These are the variables used to make predictions. In this project, our features include `Age`, `Estimated Salary`, and `Gender`.

### Linear Regression (Brief Theory)
Linear Regression is a fundamental supervised learning algorithm used primarily for regression tasks, where the goal is to predict a continuous output variable. It models the relationship between the dependent variable and one or more independent variables by fitting a linear equation to the observed data. The equation typically looks like:

$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon $

Where:
* $y$ is the dependent variable.
* $\beta_0$ is the intercept.
* $\beta_1, \beta_2, \dots, \beta_n$ are the coefficients (slopes) for each independent variable.
* $x_1, x_2, \dots, x_n$ are the independent variables.
* $\epsilon$ is the error term.

**Important Note for this Project:** While Linear Regression outputs a continuous value, our `Purchased` variable is binary (0 or 1). This project uses Linear Regression on a binary target to demonstrate the full workflow, but for true binary classification, **Logistic Regression** is generally the preferred and more appropriate model as it models the probability of an event occurring.

### Data Scaling
Data scaling is a preprocessing step that standardizes the range of independent variables or features. Features with a larger range can disproportionately influence the model's objective function. `StandardScaler` (used in this project) scales features to have a mean of 0 and a standard deviation of 1. This helps many algorithms, especially those that rely on distance calculations or gradient descent, to converge faster and perform better.

### Train-Test Split
To evaluate the model's performance on unseen data, we split our dataset into two subsets:
* **Training Set:** Used to train the machine learning model.
* **Testing Set:** Used to evaluate the model's performance on data it has not seen before. This provides an unbiased evaluation of the model's generalization ability.

## Project Structure

* `SUV_Notebook.ipynb`: The Jupyter Notebook containing all the Python code for this project.
* `suv_predictions_dataset.csv`: The dataset used for training and testing the model.

## Setup

To run this project, you'll need Python and the following libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Code Steps

The entire workflow is consolidated into the `SUV_Notebook.ipynb` file. Here, the key steps are broken down with relevant code snippets.

### Step 1: Data Collection

This step involves loading the dataset into a pandas DataFrame.

```python
import pandas as pd

try:
    df = pd.read_csv('suv_predictions_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'suv_predictions_dataset.csv' not found. Please ensure the file is in the correct directory.")
    raise # Re-raise the error for clarity
```

### Step 2: Data Analyzing

Initial analysis involves inspecting the DataFrame's columns and the first few rows to understand the data structure and types.

```python
print("\nAvailable columns in DataFrame:")
print(df.columns.tolist())

print("\nFirst 5 rows of the DataFrame:")
print(df.head())

# Identifying non-numeric columns for potential preprocessing
non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_cols:
    print(f"\nNon-numeric columns found: {non_numeric_cols}. These may need one-hot encoding or dropping.")
```

### Step 3: Data Wrangling

This crucial step involves cleaning and transforming the data, including selecting specific columns, handling categorical features (one-hot encoding), defining independent (`X`) and dependent (`y`) variables, and scaling the features.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Keep only the specified columns
required_columns = ['User ID', 'Gender', 'Age', 'Estimated Salary', 'Purchased']
# (Error handling for missing columns would be here as in the notebook)
df = df[required_columns].copy()
print(f"\nDataFrame filtered to keep only: {required_columns}")

# Define 'Purchased' as the target variable (y)
target_column = 'Purchased'
y = df[target_column]
print(f"\nDependent Variable (y) set to: '{target_column}'")

# Define Independent Variables (X)
# Drop the target column and 'User ID' (as it's just an identifier)
X_features_df = df.drop(columns=[target_column, 'User ID'], errors='ignore').copy()

# Handle categorical features ('Gender') with One-Hot Encoding
categorical_cols_for_encoding = X_features_df.select_dtypes(include='object').columns.tolist()
if categorical_cols_for_encoding:
    print(f"Applying one-hot encoding to categorical features: {categorical_cols_for_encoding}")
    X_features_df = pd.get_dummies(X_features_df, columns=categorical_cols_for_encoding, drop_first=True, dtype=int)

# Ensure all columns in X are numeric
X = X_features_df.select_dtypes(include=np.number)

# Handle any remaining NaN values (if any)
if X.isnull().any().any():
    print("Handling remaining missing values in X by imputing with mean.")
    X = X.fillna(X.mean())

print("\n--- Independent Variables (X) Head ---")
print(X.head())
print(f"Shape of X: {X.shape}")

print("\n--- Dependent Variable (y) Head ---")
print(y.head())
print(f"Shape of y: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData successfully split into training and testing sets.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully using StandardScaler.")
```

### Step 4: Training and Testing

This step involves initializing and training the Linear Regression model using the scaled training data.

```python
from sklearn.linear_model import LinearRegression

# Train the Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

print("\nLinear Regression model trained successfully!")

# Inspect the coefficients and intercept
print("\nModel Coefficients (slope for each scaled feature):")
feature_names_used = X_train.columns
coefficients_df = pd.DataFrame(regressor.coef_, feature_names_used, columns=['Coefficient'])
print(coefficients_df)

print(f"\nModel Intercept: {regressor.intercept_}")
```

### Step 5: Accuracy Check and Evaluation

Finally, we make predictions on the unseen test data, convert the continuous predictions to binary (since our target is binary), and then evaluate the model using accuracy and other relevant classification metrics.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Make Predictions on the Test Set
y_pred_continuous = regressor.predict(X_test_scaled)

# Convert continuous predictions to binary (0 or 1) using a threshold
# For Linear Regression on a binary target, a common threshold is 0.5
y_pred_binary = (y_pred_continuous >= 0.5).astype(int)

print("\n--- Sample Predictions (Actual vs. Predicted Binary) ---")
predictions_df = pd.DataFrame({'Actual': y_test.head(10), 'Predicted_Continuous': y_pred_continuous[:10], 'Predicted_Binary': y_pred_binary[:10]})
print(predictions_df)

# Calculate Accuracy Score
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))

# Display Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Calculate ROC AUC Score (uses continuous predictions as scores for a binary target)
roc_auc = roc_auc_score(y_test, y_pred_continuous)
print(f"\nROC AUC Score: {roc_auc:.4f}")

print("\nModel evaluation complete!")
print("Note: For binary classification problems like 'Purchased', Logistic Regression is generally preferred.")
```

## Conclusion

This project successfully demonstrates a complete machine learning pipeline from data loading and preprocessing to model training and evaluation using Linear Regression. While Linear Regression provides a foundational understanding, the nature of the `Purchased` target variable (binary) indicates that classification algorithms like Logistic Regression would provide a more direct and interpretable solution for predicting purchase probability.

---
