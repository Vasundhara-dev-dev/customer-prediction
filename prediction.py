import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScalar
from sklearn.linear_model import LinearRegression

df = pd.read_excel("online_retail_II (1).xlsx")
df.head()

df = df[df['InvoiceNo'].notnull()]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df.dropna(inplace = True)

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days = 1)
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 'InvoiceNo': 'nunique', 'TotalSum': lambda x: (df.loc[x.index, 'Quantity'] * df.loc[x.index, 'UnitPrice']).sum()}).rename(columns = {'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'MonetaryValue'}).reset_index()

scalar = StandardScalar()
rfm_scaled = scalar.fit_transform(rfm[['Recency', 'Frequency', 'MonetaryValue']])
kmeans = KMeans(n_clusters = 4)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

df['TotalSum'] = df['Quantity'] * df['UnitPrice']
customer_lifetime = df.groupby('CustomerID').agg({'InvoiceDate': [np.min, np.max], 'InvoiceNo': 'nunique', 'TotalSum': 'sum'})
customer_lifetime.columns = ['FirstPurchase', 'LastPurchase', 'Frequency', 'Monetary']
customer_lifetime['Age'] = (customer_lifetime['LastPurchase'] - customer_lifetime['FirstPurchase']).dt.days
customer_lifetime['CLV'] = (customer_lifetime['Monetary'] / customer_lifetime['Frequency']) * customer_lifetime['Frequency'] * (customer_lifetime['Age'] / 365)

x = customer_lifetime[['Frequency', 'Monetary', 'Age']]
y = customer_lifetime['CLV']

model = LinearRegression()
model.fit(x, y)

predicted_clv = model.predict(x)

sns.histplot(customer_lifetime['CLV'], bins = 50)
plt.title("Customer Lifetime Value Prediction")
plt.xlabel("CLV")
plt.show()
