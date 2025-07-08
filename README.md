Customer Lifetime Value (CLV) Prediction

This project predicts the customer lifetime value using historical purchase behaviour from the given dataset. 

Goal:
The goal is to estimate how valuable each customer will be in the future, based on:
1. Recency: Days since the last purchase
2. Frequency: Total no. of purchases
3. Monetary: Total amount spent

Steps:

1. Data Cleaning:
   a. Removed cancelled/returned orders (invoices starting with 'C')
   b. Dropped missing 'Customer ID'
   c. Filtered out negative/zero quantity or price

2. Feature Engineering:
   a. Created 'TotalPrice = Quantity * Price'
   b. Calculated 'Recency', 'Frequency' and 'Monetary' per customer
   c. Used the above to build the RFM table

3. Modeling:
   a. Normalized features using 'StandardScaler'
   b. Trained a 'LinearRegression' model
   c. Predicted CLV based on RFM inputs

4. Visualization: Plot distribution of predicted CLV values
