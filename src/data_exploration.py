
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/waste_data.csv")

# Dataset information
print("Dataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 1. Waste Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Waste_Type")
plt.title("Waste Type Distribution")
plt.tight_layout()
plt.savefig("graphs/waste_distribution.png")
plt.close()

# 2. Collection Requirement Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Collection_Requirement")
plt.title("Collection Requirement Distribution")
plt.tight_layout()
plt.savefig("graphs/collection_distribution.png")
plt.close()

# 3. Waste Weight Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Waste_Weight", kde=True)
plt.title("Waste Weight Distribution")
plt.tight_layout()
plt.savefig("graphs/waste_weight.png")
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 7))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png")
plt.close()

# 5. Moisture by Waste Type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Waste_Type", y="Moisture")
plt.title("Moisture by Waste Type")
plt.tight_layout()
plt.savefig("graphs/moisture_by_waste.png")
plt.close()

# 6. Bin Fill Level by Collection Requirement
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Collection_Requirement",
    y="Bin_Fill_Level"
)
plt.title("Bin Fill Level by Collection Requirement")
plt.tight_layout()
plt.savefig("graphs/bin_fill_collection.png")
plt.close()

print("\nAll six graphs saved successfully in graphs folder!")
