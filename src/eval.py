
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

os.makedirs("output", exist_ok=True)

# =================================
# LOAD TEST DATA
# =================================

# Waste classification
waste_X_test = pd.read_csv("output/waste_X_test.csv")
waste_y_test = pd.read_csv(
    "output/waste_y_test.csv"
).values.ravel()

waste_X_test_scaled = np.load(
    "output/waste_X_test_scaled.npy"
)

# Collection prediction
collection_X_test = pd.read_csv("output/collection_X_test.csv")
collection_y_test = pd.read_csv(
    "output/collection_y_test.csv"
).values.ravel()

collection_X_test_scaled = np.load(
    "output/collection_X_test_scaled.npy"
)


# =================================
# LOAD MODELS
# =================================

models = {
    "Waste Decision Tree": (
        "models/waste_decision_tree.pkl",
        waste_X_test
    ),

    "Waste Random Forest": (
        "models/waste_random_forest.pkl",
        waste_X_test
    ),

    "Waste SVM": (
        "models/waste_svm.pkl",
        waste_X_test_scaled
    ),

    "Collection Decision Tree": (
        "models/collection_decision_tree.pkl",
        collection_X_test
    ),

    "Collection Random Forest": (
        "models/collection_random_forest.pkl",
        collection_X_test
    ),

    "Collection SVM": (
        "models/collection_svm.pkl",
        collection_X_test_scaled
    )
}


# =================================
# MODEL EVALUATION
# =================================

results = []

for model_name, (model_path, X_test) in models.items():

    model = joblib.load(model_path)

    if "Waste" in model_name:
        y_test = waste_y_test
        encoder = joblib.load("models/waste_encoder.pkl")
    else:
        y_test = collection_y_test
        encoder = joblib.load("models/collection_encoder.pkl")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n=================================")
    print(model_name)
    print("=================================")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2)
    })


# =================================
# SAVE ACCURACY RESULTS
# =================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "output/model_accuracy.csv",
    index=False
)

print("\n=================================")
print("MODEL ACCURACY COMPARISON")
print("=================================")

print(results_df.to_string(index=False))

print("\nEvaluation completed successfully!")
print("Results saved in output/model_accuracy.csv")
