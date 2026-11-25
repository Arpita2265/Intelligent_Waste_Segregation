
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

os.makedirs("models", exist_ok=True)

# =================================
# LOAD PREPROCESSED DATA
# =================================

# Waste classification
waste_X_train = pd.read_csv("output/waste_X_train.csv")
waste_y_train = pd.read_csv("output/waste_y_train.csv").values.ravel()

waste_X_train_scaled = np.load(
    "output/waste_X_train_scaled.npy"
)

# Collection prediction
collection_X_train = pd.read_csv("output/collection_X_train.csv")
collection_y_train = pd.read_csv(
    "output/collection_y_train.csv"
).values.ravel()

collection_X_train_scaled = np.load(
    "output/collection_X_train_scaled.npy"
)


# =================================
# WASTE CLASSIFICATION MODELS
# =================================

print("\nTraining Waste Classification Models...")

# 1. Decision Tree
waste_dt = DecisionTreeClassifier(random_state=42)
waste_dt.fit(waste_X_train, waste_y_train)

joblib.dump(waste_dt, "models/waste_decision_tree.pkl")
print("Waste Decision Tree saved!")

# 2. Random Forest
waste_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
waste_rf.fit(waste_X_train, waste_y_train)

joblib.dump(waste_rf, "models/waste_random_forest.pkl")
print("Waste Random Forest saved!")

# 3. SVM
waste_svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
waste_svm.fit(waste_X_train_scaled, waste_y_train)

joblib.dump(waste_svm, "models/waste_svm.pkl")
print("Waste SVM saved!")


# =================================
# COLLECTION PREDICTION MODELS
# =================================

print("\nTraining Collection Prediction Models...")

# 1. Decision Tree
collection_dt = DecisionTreeClassifier(random_state=42)
collection_dt.fit(collection_X_train, collection_y_train)

joblib.dump(
    collection_dt,
    "models/collection_decision_tree.pkl"
)
print("Collection Decision Tree saved!")

# 2. Random Forest
collection_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
collection_rf.fit(collection_X_train, collection_y_train)

joblib.dump(
    collection_rf,
    "models/collection_random_forest.pkl"
)
print("Collection Random Forest saved!")

# 3. SVM
collection_svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
collection_svm.fit(
    collection_X_train_scaled,
    collection_y_train
)

joblib.dump(
    collection_svm,
    "models/collection_svm.pkl"
)
print("Collection SVM saved!")


# =================================
# COMPLETION
# =================================

print("\nAll 6 Models Trained Successfully!")
