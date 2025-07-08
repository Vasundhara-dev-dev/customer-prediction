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
