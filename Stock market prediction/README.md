# Stock Market Prediction Project Documentation

## Introduction

This project analyzes historical stock market data for the year 2022 and aims to predict if a stock's closing price will increase by at least 5% within the next 5 days (open market days). The analysis focuses on data for 31 different technology companies represented by their ticker symbols.

Data : https://www.kaggle.com/datasets/luisandresgarcia/stock-market-prediction

## Approach and Methodologies:

### Data Exploration: 
The dataset comprises 7781 rows and 1285 columns, containing stock market data for the year 2022 across 31 different tickers. The initial step involved exploring the dataset to understand its structure, features, and contents. 

### Feature Selection: 
Given the vast number of features, a careful selection process was undertaken to identify the most relevant ones for predictive modeling. Features such as 'open', 'high', 'low', 'close', and 'volume' were chosen due to their significance in stock market analysis.

### Data Filtering: 
Since the dataset contains data for multiple years and multiple tickers, filtering was performed to isolate the data for the year 2022 and for each individual ticker. This ensured that the analysis and modeling focused specifically on the relevant timeframe and tickers.

### Visualization: 
Visual exploration of the selected features was conducted to identify trends, patterns, and relationships. Time series plots were created to visualize the behavior of stock prices over time for each ticker, allowing for insights into price movements and volatility.

### Model Selection: 
Two predictive models were chosen for forecasting future stock values: logistic regression and random forest classifier. These models were selected based on their suitability for binary classification tasks and their ability to handle complex relationships in the data.

### Model Training and Evaluation: 
The dataset was split into training and testing sets, and the selected models were trained using the training data. Model performance was evaluated using accuracy score as the primary metric, with additional classification metrics such as precision, recall, and F1-score calculated for further insight.

## Insights Gained from the Stock Market Dataset:

### Yearly Data Analysis: 
By focusing on the data specifically from the year 2022, insights were gained into the stock market behavior and trends during that particular timeframe. This allowed for a more targeted analysis and modeling approach.

### Ticker-specific Analysis: 
Analyzing data for each individual ticker provided valuable insights into the performance and behavior of different stocks. Patterns and trends specific to each ticker were identified, aiding in understanding the unique characteristics of each stock.

### Feature Importance: 
The process of feature selection revealed the importance of certain variables such as 'open', 'high', 'low', 'close', and 'volume' in predicting stock price movements. These features were deemed significant based on their potential impact on the target variable.

### Model Performance: 
Both logistic regression and random forest classifier exhibited promising performance in predicting future stock values. The accuracy scores obtained from these models indicate their capability to make accurate predictions based on the selected features.

## Explanations for Chosen Predictive Models:

### Logistic Regression: 
Logistic regression was chosen for its simplicity, interpretability, and efficiency in binary classification tasks. It models the probability of a binary outcome based on one or more predictor variables, making it suitable for predicting stock price movements (e.g., increase or decrease).

### Random Forest Classifier: 
The random forest classifier was selected for its ability to handle large datasets with high dimensionality and capture nonlinear relationships between features and the target variable. By combining multiple decision trees, it can provide robust predictions while mitigating the risk of overfitting.

In summary, the combination of data exploration, feature selection, visualization, and predictive modeling techniques provided valuable insights into the behavior of the stock market dataset for the year 2022. The selected models offer a framework for forecasting future stock values and aiding decision-making in the dynamic and complex stock market environment.
