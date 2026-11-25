
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/waste_data.csv")

# Check and remove duplicates
print("Missing values:\n", df.isnull().sum())
print("\nDuplicate records:", df.duplicated().sum())

df = df.drop_duplicates().copy()


# =================================
# FILL MISSING VALUES
# =================================

numeric_columns = [
    "Waste_Weight",
    "Moisture",
    "Recyclability",
    "Material_Density",
    "Bin_Fill_Level",
    "Days_Since_Collection"
]

categorical_columns = [
    "Waste_Type",
    "Collection_Requirement"
]

# Convert numerical columns into numeric format
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill numerical missing values
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# =================================
# WASTE CLASSIFICATION
# =================================

waste_features = [
    "Waste_Weight",
    "Moisture",
    "Recyclability",
    "Material_Density",
    "Bin_Fill_Level",
    "Days_Since_Collection"
]

X_waste = df[waste_features]
y_waste = df["Waste_Type"]

waste_encoder = LabelEncoder()
y_waste_encoded = waste_encoder.fit_transform(y_waste)

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_waste,
    y_waste_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_waste_encoded
)

# Scale data for SVM
waste_scaler = StandardScaler()

Xw_train_scaled = waste_scaler.fit_transform(Xw_train)
Xw_test_scaled = waste_scaler.transform(Xw_test)

# Save waste datasets
Xw_train.to_csv("output/waste_X_train.csv", index=False)
Xw_test.to_csv("output/waste_X_test.csv", index=False)

pd.DataFrame(yw_train).to_csv(
    "output/waste_y_train.csv", index=False
)

pd.DataFrame(yw_test).to_csv(
    "output/waste_y_test.csv", index=False
)

np.save("output/waste_X_train_scaled.npy", Xw_train_scaled)
np.save("output/waste_X_test_scaled.npy", Xw_test_scaled)

joblib.dump(waste_scaler, "models/waste_scaler.pkl")
joblib.dump(waste_encoder, "models/waste_encoder.pkl")


# =================================
# COLLECTION PREDICTION
# =================================

collection_features = [
    "Waste_Weight",
    "Moisture",
    "Recyclability",
    "Material_Density",
    "Bin_Fill_Level",
    "Days_Since_Collection",
    "Waste_Type"
]

X_collection = pd.get_dummies(
    df[collection_features],
    columns=["Waste_Type"],
    dtype=int
)

y_collection = df["Collection_Requirement"]

collection_encoder = LabelEncoder()

y_collection_encoded = collection_encoder.fit_transform(
    y_collection
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_collection,
    y_collection_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_collection_encoded
)

# Scale data for SVM
collection_scaler = StandardScaler()

Xc_train_scaled = collection_scaler.fit_transform(Xc_train)
Xc_test_scaled = collection_scaler.transform(Xc_test)

# Save collection datasets
Xc_train.to_csv("output/collection_X_train.csv", index=False)
Xc_test.to_csv("output/collection_X_test.csv", index=False)

pd.DataFrame(yc_train).to_csv(
    "output/collection_y_train.csv", index=False
)

pd.DataFrame(yc_test).to_csv(
    "output/collection_y_test.csv", index=False
)

np.save("output/collection_X_train_scaled.npy", Xc_train_scaled)
np.save("output/collection_X_test_scaled.npy", Xc_test_scaled)

joblib.dump(collection_scaler, "models/collection_scaler.pkl")
joblib.dump(collection_encoder, "models/collection_encoder.pkl")


# =================================
# COMPLETION
# =================================

print("\nPreprocessing completed successfully!")

print("\nWaste Training Records:", len(Xw_train))
print("Waste Testing Records:", len(Xw_test))

print("\nCollection Training Records:", len(Xc_train))
print("Collection Testing Records:", len(Xc_test))
