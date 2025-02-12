# Gold Price Prediction Using Machine Learning

## Overview
This project focuses on predicting gold prices (denoted as GLD) using machine learning techniques. The analysis involves various financial indicators such as stock prices and currency exchange rates to predict the value of gold based on historical data.

## Objectives
The primary objectives of this notebook are to:
1. **Build a Machine Learning Model**: Create a model that can accurately predict GLD prices based on other financial metrics.
2. **Data Insights**: Obtain insights from the data using the Pandas library.
3. **Correlation Analysis**: Analyze the correlation of other features with the GLD price.
4. **Model Training and Evaluation**: Split the data into training and test sets for model training and evaluate its performance.

## About the Data
### Data Overview
The dataset utilized for this analysis is structured in a Comma-Separated Value (CSV) file format, containing:
- **Rows**: 2290
- **Columns**: 7
  - **Date**: Date in MM/DD/YYYY format (object type).
  - **SPX**: The Standard and Poor's 500 Index (float).
  - **GLD**: SPDR Gold Shares ETF price (float).
  - **USO**: United States Oil Fund price (float).
  - **SLV**: iShares Silver Trust price (float).
  - **EUR/USD**: Currency pair value for Euro to USD (float).

### Data Source
*For the data source link, please insert the relevant link here.*

### Data Types
The dataset contains five numerical columns and one date column. 

### Import Required Libraries
To begin working with the data, we first need to import the necessary libraries.

```python
import pandas as pd                                          # Load data & perform basic operations
import numpy as np                                           # Numpy for numerical operations
import matplotlib.pyplot as plt                              # Matplotlib for plotting graphs
import seaborn as sns                                        # Seaborn for enhanced data visualization
from sklearn.model_selection import train_test_split         # Split data into training and testing sets
from sklearn.ensemble import RandomForestRegressor           # Import the Random Forest Regression Model
from sklearn import metrics                                  # Performance metrics for model evaluation
```

## Data Collection & Processing
### Loading the Dataset
We load the `gld_price_data.csv` file and review the structure of the data.

```python
# Load the data from CSV file
gold_data = pd.read_csv("gld_price_data.csv")

# Display the first five and last five rows of the dataset
print(gold_data.head())
print(gold_data.tail())
```

### Understanding the Data
After loading, we analyze the fundamental aspects of the dataset.

- **Shape of the Dataset**: To get the number of rows and columns:
```python
print(gold_data.shape)  # Output: (2290, 6)
```

- **Basic Information**:
```python
print(gold_data.info())  # Provides info on data types and non-null counts
```

- **Missing Values**:
```python
print(gold_data.isnull().sum())  # Check for missing values in each column
```

- **Statistical Summary**:
```python
print(gold_data.describe())  # Get descriptive statistics
```

### Data Column Descriptions
- **SPX**: Represents the S&P 500 index, which tracks the performance of 500 large companies listed on stock exchanges in the United States.
- **GLD**: Represents the price of SPDR Gold Shares, part of the SPDR family of exchange-traded funds (ETFs).
- **USO**: Represents the United States Oil Fund LP (USO), which is a traded security for oil investment.
- **SLV**: Represents the iShares Silver Trust, an ETF tracking the silver price.
- **EUR/USD**: Represents the exchange rate between Euros and U.S. Dollars.

## Correlation Analysis
We need to understand how GLD prices correlate with other financial variables. 

### Calculating Correlation
To identify relationships, we compute a correlation matrix.

```python
# Convert the 'Date' column to datetime type for proper analysis
gold_data['Date'] = pd.to_datetime(gold_data['Date'], errors='coerce')

# Compute the correlation matrix excluding non-numeric columns
correlation = gold_data.select_dtypes(include=['float64', 'int64']).corr()

# Displaying the correlation matrix
print("Correlation Matrix:")
print(correlation)

# Visualizing the correlation using a heatmap
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')
```

![heatmap](https://github.com/user-attachments/assets/3036b824-9e27-4db7-98b0-c6b6a553813f)


### Interpretation of Correlations
- A positive correlation indicates that as one variable increases, the other tends to increase.
- A negative correlation indicates that as one variable increases, the other tends to decrease.
- Zero correlation indicates no relationship between the variables.

From the correlation matrix output, we observe:
- **GLD (Gold)** and **SLV (Silver)** show a strong correlation (approximately +0.87).
- **GLD** has a negligible positive correlation with **SPX** (about +0.05).
- **GLD** has a mild negative correlation with **USO** (approximately -0.19).

## Data Visualization
Before training the model, it is useful to visualize the distribution of GLD prices.

```python
# checking the distribution of the GLD Price using histplot
sns.histplot(gold_data["GLD"], kde=True, color="Green")
plt.show()
```

![histplot](https://github.com/user-attachments/assets/653b9c0d-6d9f-485f-8121-b825db0c6c63)


The visualization indicates that most GLD prices are concentrated around the value of 120, with fewer occurrences at the extremes.

## Feature Selection
### Splitting Features and Target Variable
Here we set the target variable (Y) as GLD prices and the feature set (X) as the other stock prices excluding GLD and the date.

```python
# Define features (X) and target variable (Y)
X = gold_data.drop(["Date", "GLD"], axis=1)  # All columns except 'Date' and 'GLD'
Y = gold_data["GLD"]  # Target variable GLD

print(X.head())  # Display the features
print(Y.head())  # Display the target variable
```

## Train-Test Split
We split the dataset into training and test sets to evaluate the model's performance.

```python
# Dividing the data into training and testing sets (80% - 20% split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f'Training set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')
```

## Model Training
### Random Forest Regressor
We utilize the Random Forest Regressor, which is an ensemble method that combines several decision trees to increase the predictive performance.

```python
# Initializing the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model with training data
regressor.fit(X_train, Y_train)
print("Model training completed.")
```

## Model Evaluation
After training the model, we predict the GLD values using the test data and evaluate the model's accuracy.

```python
# Making predictions on the test set
predictions = regressor.predict(X_test)

# Evaluating the model using R-squared metric
error_score = metrics.r2_score(Y_test, predictions)
print("R-squared score: ", error_score)
```

### Visual Comparison of Actual vs. Predicted Values
We visualize how well the model's predictions reflect the actual GLD prices.

```python
# Plotting actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(Y_test.values, color='blue', label='Actual Values')  # Actual values from the test set
plt.plot(predictions, color='green', label='Predicted Values')  # Predicted values
plt.title("Comparison of Actual vs Predicted GLD Prices")
plt.xlabel("Test Sample Index")
plt.ylabel("GLD Price")
plt.legend()
plt.show()
```

## Conclusion
The analysis detailed in this notebook demonstrates the application of machine learning to predict gold prices using historical data from various financial indicators. The Random Forest Regressor achieved an R-squared score of approximately **0.989**, indicating a high level of predictive accuracy. This suggests that our model is capable of effectively forecasting GLD prices based on the input features.

Future enhancements could include:
- Exploring additional features that could improve prediction accuracy.
- Testing other machine learning algorithms for comparison.
- Continuously updating the model with new data as it becomes available.
