# Daily Demand Forecasting using Support Vector Regression (SVR)
This repository contains a Python script for predicting daily demand forecasting orders using Support Vector Regression (SVR). SVR is a supervised machine learning algorithm that can be used for regression tasks. In this project, it's applied to forecast the total daily orders based on various features.

# Dataset
The dataset used in this project is loaded from an Excel file named "Daily_Demand_Forecasting_Orders_reg.xls". It contains the following columns:

Day of the week: Numeric representation of the day (1 to 7).
Banking orders (1, 2, 3): Numerical values representing different banking orders.
Target (Total orders): Total daily orders (the target variable).
# Data Preprocessing
Checked for and handled missing values.
Checked for and removed outliers using the Interquartile Range (IQR) method.
Dropped columns with less than 40% correlation with the target variable: 'Week of the month', 'Fiscal sector orders', 'Orders from the traffic controller sector'.
Dropped columns with high multicollinearity: 'Non-urgent order', 'Urgent order', 'Order type A', 'Order type B', 'Order type C'.
# Model Building and Optimization
Split the data into training and testing sets (80% training, 20% testing).
Implemented a base SVR model and evaluated its performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.
Conducted hyperparameter tuning using Randomized Search to find the best combination of hyperparameters.
Built an optimized SVR model using the best hyperparameters obtained from the Randomized Search.
# Results
The optimized SVR model achieved a higher accuracy compared to the base model.
The most significant predictor for total daily orders is found to be 'Banking orders (2)', as it has the highest positive impact on the prediction accuracy.
