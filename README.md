# Real-Estate Price Prediction

Real-Estate Price Prediction is a machine learning-based project developed in Python. It utilizes popular libraries such as NumPy, Pandas, SciPy, and scikit-learn to build a predictive model for estimating real estate prices. The dataset used in this project is relatively small and fetched from the internet.

## Overview

The goal of this project is to develop a predictive model that can accurately estimate real estate prices based on various features such as location, size, number of bedrooms, etc. The model is trained on the provided dataset and then used to make predictions on new data.

## Libraries Used

- **NumPy**: For numerical computing and handling arrays.
- **Pandas**: For data manipulation and analysis.
- **SciPy**: For scientific computing and statistical functions.
- **scikit-learn**: For building and evaluating machine learning models.

## Dataset

The dataset used in this project is fetched from the internet and contains information about various real estate properties. It includes features such as:
- Location
- Size
- Number of bedrooms
- Price

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KalimMulani/Real-Estate-Price-Prediction.git
    ```
    
## Usage

1. **Data Preparation**:
    - Clean and preprocess the dataset, handle missing values, and encode categorical variables if necessary.

2. **Feature Engineering**:
    - Extract relevant features from the dataset and perform feature scaling or normalization if required.

3. **Model Training**:
    - Utilize scikit-learn to train machine learning models such as linear regression, decision trees, or ensemble methods on the prepared dataset.

4. **Model Evaluation**:
    - Evaluate the performance of the trained models using appropriate metrics such as mean squared error, mean absolute error, etc.

5. **Prediction**:
    - Use the trained model to make predictions on new real estate data.

## Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('real_estate_data.csv')

# Split data into features and target variable
X = data.drop('Price', axis=1)
y = data['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

```

#License
This project is licensed under the MIT License.
