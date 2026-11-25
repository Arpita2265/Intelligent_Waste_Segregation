
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Intelligent Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Intelligent Waste Segregation")
st.subheader("Waste Classification and Collection Prediction")

# =================================
# LOAD MODELS
# =================================

@st.cache_resource
def load_models():

    models = {
        "Decision Tree": {
            "waste": joblib.load("models/waste_decision_tree.pkl"),
            "collection": joblib.load("models/collection_decision_tree.pkl")
        },

        "Random Forest": {
            "waste": joblib.load("models/waste_random_forest.pkl"),
            "collection": joblib.load("models/collection_random_forest.pkl")
        },

        "SVM": {
            "waste": joblib.load("models/waste_svm.pkl"),
            "collection": joblib.load("models/collection_svm.pkl")
        }
    }

    waste_scaler = joblib.load("models/waste_scaler.pkl")
    collection_scaler = joblib.load("models/collection_scaler.pkl")

    waste_encoder = joblib.load("models/waste_encoder.pkl")
    collection_encoder = joblib.load("models/collection_encoder.pkl")

    return (
        models,
        waste_scaler,
        collection_scaler,
        waste_encoder,
        collection_encoder
    )


models, waste_scaler, collection_scaler, waste_encoder, collection_encoder = load_models()


# =================================
# SIDEBAR
# =================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Waste Prediction",
        "Collection Prediction",
        "Model Performance"
    ]
)


# =================================
# INPUT FIELDS
# =================================

def get_inputs():

    col1, col2 = st.columns(2)

    with col1:
        weight = st.number_input("Waste Weight (kg)", min_value=0.0, value=2.0)
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=30.0)
        recyclability = st.number_input("Recyclability (%)", min_value=0.0, max_value=100.0, value=50.0)

    with col2:
        density = st.number_input("Material Density", min_value=0.0, value=1.0)
        fill_level = st.number_input("Bin Fill Level (%)", min_value=0.0, max_value=100.0, value=50.0)
        days = st.number_input("Days Since Collection", min_value=0, value=3)

    return [
        weight,
        moisture,
        recyclability,
        density,
        fill_level,
        days
    ]


# =================================
# HOME PAGE
# =================================

if page == "Home":

    st.header("Welcome")

    st.write(
        "This project uses Machine Learning to classify waste "
        "and predict collection requirements."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithms", "3")
    col2.metric("Prediction Tasks", "2")
    col3.metric("Trained Models", "6")

    st.info(
        "Select a page from the sidebar to start making predictions."
    )


# =================================
# WASTE PREDICTION
# =================================

elif page == "Waste Prediction":

    st.header("Waste Classification")

    algorithm = st.selectbox(
        "Select Algorithm",
        ["Decision Tree", "Random Forest", "SVM"]
    )

    inputs = get_inputs()

    if st.button("Predict Waste Type"):

        input_df = pd.DataFrame(
            [inputs],
            columns=[
                "Waste_Weight",
                "Moisture",
                "Recyclability",
                "Material_Density",
                "Bin_Fill_Level",
                "Days_Since_Collection"
            ]
        )

        if algorithm == "SVM":
            input_data = waste_scaler.transform(input_df)
        else:
            input_data = input_df

        prediction = models[algorithm]["waste"].predict(input_data)

        result = waste_encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Waste Type: {result}")


# =================================
# COLLECTION PREDICTION
# =================================

elif page == "Collection Prediction":

    st.header("Collection Requirement Prediction")

    algorithm = st.selectbox(
        "Select Algorithm",
        ["Decision Tree", "Random Forest", "SVM"]
    )

    inputs = get_inputs()

    waste_type = st.selectbox(
        "Select Waste Type",
        waste_encoder.classes_
    )

    if st.button("Predict Collection Requirement"):

        input_df = pd.DataFrame(
            [[
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5]
            ]],
            columns=[
                "Waste_Weight",
                "Moisture",
                "Recyclability",
                "Material_Density",
                "Bin_Fill_Level",
                "Days_Since_Collection"
            ]
        )

        input_df["Waste_Type"] = waste_type

        input_df = pd.get_dummies(
            input_df,
            columns=["Waste_Type"],
            dtype=int
        )

        feature_columns = models[algorithm]["collection"].feature_names_in_

        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        if algorithm == "SVM":
            input_data = collection_scaler.transform(input_df)
        else:
            input_data = input_df

        prediction = models[algorithm]["collection"].predict(input_data)

        result = collection_encoder.inverse_transform(prediction)[0]

        st.success(f"Collection Requirement: {result}")


# =================================
# MODEL PERFORMANCE
# =================================

elif page == "Model Performance":

    st.header("Model Accuracy Comparison")

    results = pd.read_csv("output/model_accuracy.csv")

    st.dataframe(results, use_container_width=True)

    st.bar_chart(
        results.set_index("Model")["Accuracy"]
    )

    st.subheader("Accuracy Graph")

    st.image(
        "graphs/model_accuracy_comparison.png",
        use_container_width=True
    )

    st.caption("Accuracy is based on the current test dataset.")


# =================================
# FOOTER
# =================================

st.sidebar.markdown("---")
st.sidebar.caption("Intelligent Waste Management System")
