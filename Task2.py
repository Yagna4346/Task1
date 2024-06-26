import pandas as pd
import numpy as np
df = pd.read_csv('abalone.csv')
print("First few rows of the DataFrame:")
print(df.head())
print("\nSummary statistics of the DataFrame:")
print(df.describe())
print("\nInformation about the DataFrame:")
print(df.info())
print("few last rows of the DataFrame:")
print(df.tail())
filtered_df = df[df['Rings'] > 5]
print("\nFiltered DataFrame where 'Rings' > 5:")
print(filtered_df)
# Handle missing values
df_dropped = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(df_dropped)
df_filled = df.fillna(0)
print("\nDataFrame after filling missing values with 0:")
print(df_filled)
numeric_df = df_filled.select_dtypes(include=[np.number])
#statistics
mean_val= numeric_df.mean()
median_val= numeric_df.median()
std_val= numeric_df.std()
print("\nMean values of each column:")
print(mean_val)
print("\nMedian values of each column:")
print(median_val)
print("\nStandard deviation of each column:")
print(std_val)
