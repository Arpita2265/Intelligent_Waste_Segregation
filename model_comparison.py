
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("graphs", exist_ok=True)

# Load accuracy results
df = pd.read_csv("output/model_accuracy.csv")

# Display accuracy
print(df)

# =================================
# MODEL ACCURACY COMPARISON
# =================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "graphs/model_accuracy_comparison.png"
)

plt.show()

print("\nAccuracy graph saved successfully!")
