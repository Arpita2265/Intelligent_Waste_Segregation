
import pandas as pd

df = pd.read_csv("dataset/waste_data.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nWaste Type Distribution:")
print(df["Waste_Type"].value_counts())

print("\nCollection Requirement Distribution:")
print(df["Collection_Requirement"].value_counts())