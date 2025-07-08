import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_excel("online_retail_II (1).xlsx")
df.head()

df = df.dropna(subset = ['Customer ID'])
df = df[~df['Invoice'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['Price']

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days = 1)
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 'Invoice': 'nunique', 'TotalPrice': 'sum'}).reset_index()

rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'MonetaryValue']

x = rfm[['Recency', 'Frequency', 'MonetaryValue']]
y = rfm['MonetaryValue']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = LinearRegression()
model.fit(x_scaled, y)

rfm['Predicted CLV'] = model.predict(x_scaled)

plt.figure(figsize = (10, 6))
sns.histplot(rfm['Predicted CLV'], bins = 50, kde = True, color = 'green')
plt.title('Customer Lifetime Value Prediction')
plt.xlabel('Predicted CLV')
plt.ylabel('No. of Customers')
plt.show()
