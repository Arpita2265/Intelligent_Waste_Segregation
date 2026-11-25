
# ♻️ Intelligent Waste Segregation and Collection Prediction Using Machine Learning

## 📌 1. Project Overview

The **Intelligent Waste Segregation and Collection Prediction System** is a Machine Learning-based project designed to classify different types of waste and predict the collection requirement.

The system uses waste-related information such as weight, moisture, recyclability, material density, bin fill level, and days since collection to make predictions.

It performs two main tasks:

- **Waste Type Classification:** Identifies the type of waste as Plastic, Paper, Metal, Glass, or Organic.
- **Collection Requirement Prediction:** Predicts whether waste collection is required at a Low, Medium, or High level.

A Streamlit web application allows users to enter waste details and view predictions using different trained machine learning models.

---

## 🎯 2. Problem Statement

Improper waste segregation and inefficient waste collection create environmental pollution and waste-management problems.

Manual waste classification and collection planning can be time-consuming and inefficient.

This project aims to use Machine Learning techniques to automate waste classification and predict collection requirements based on waste characteristics.

---

## 🎯 3. Project Objectives

- Classify waste into different categories.
- Predict waste collection requirements.
- Apply Machine Learning algorithms to waste management.
- Compare the performance of three Machine Learning models.
- Develop an interactive web application using Streamlit.
- Display model performance through graphs and evaluation reports.

---

## 📊 4. Dataset Description

The project uses a CSV dataset named `waste_data.csv`.

**Dataset Information:**

- Total Records: 30
- Total Features: 8
- File Format: CSV

### Input Features

| Feature | Description |
|---|---|
| Waste_Weight | Weight of the waste |
| Moisture | Moisture content of waste |
| Recyclability | Recyclability information |
| Material_Density | Density of the waste material |
| Bin_Fill_Level | Percentage of bin filling |
| Days_Since_Collection | Number of days since the last collection |

### Target Variables

**1. Waste Type Classification**

- Plastic
- Paper
- Metal
- Glass
- Organic

**2. Collection Requirement Prediction**

- Low
- Medium
- High

---

## 🛠️ 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning algorithms |
| Joblib | Saving trained models |
| Streamlit | Web application |

---

## 🔄 6. Project Workflow

The project follows these steps:

1. Python Environment Setup
2. Dataset Collection and Loading
3. Data Exploration and Analysis
4. Data Preprocessing
5. Feature Scaling and Encoding
6. Model Training
7. Model Evaluation
8. Model Performance Visualization
9. Streamlit Web Application Development

---

## 🤖 7. Machine Learning Algorithms

Exactly three Machine Learning algorithms are used for both prediction tasks.

### 1. Decision Tree

A supervised Machine Learning algorithm that makes predictions using a tree-like structure of decision rules.

### 2. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

### 3. Support Vector Machine (SVM)

A supervised Machine Learning algorithm that finds an optimal decision boundary to classify data.

### Total Trained Models: 6

| Prediction Task | Models |
|---|---|
| Waste Classification | Decision Tree, Random Forest, SVM |
| Collection Prediction | Decision Tree, Random Forest, SVM |

---

## 📈 8. Model Evaluation Results

The trained models are evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

### Model Accuracy Comparison

| Model | Accuracy |
|---|---:|
| Waste Decision Tree | 100% |
| Waste Random Forest | 100% |
| Waste SVM | 44.44% |
| Collection Decision Tree | 100% |
| Collection Random Forest | 100% |
| Collection SVM | 88.89% |

### Best Performing Models

- Waste Classification: Decision Tree and Random Forest
- Collection Prediction: Decision Tree and Random Forest

**Note:** The dataset contains only 30 records, with 9 test records per prediction task. Therefore, these accuracy results are preliminary and do not guarantee the same performance on larger or unseen datasets.

---

## 📊 9. Output Graphs

The project generates graphs for data analysis and model comparison.

### Dataset Visualization

- Waste Type Distribution
- Collection Requirement Distribution
- Waste Weight Distribution
- Correlation Heatmap
- Moisture by Waste Type
- Bin Fill Level vs Collection Requirement

### Model Visualization

- Model Accuracy Comparison

All generated graphs are saved inside the `graphs/` folder.

---

## 🌐 10. Web Application

The project includes an interactive web application developed using Streamlit.

### Application Features

- Home Page
- Waste Type Prediction
- Collection Requirement Prediction
- Model Selection
- Model Performance Comparison
- Prediction Results

Users can select a trained model, enter waste details, and view the predicted result.

### Run the Application

Open the terminal in your project folder and execute:

```bash
streamlit run main.py
```

The application will open in your web browser.

---

## 📁 11. Project Folder Structure

```text
Intelligent_Waste_Segregation/
│
├── dataset/
│   └── waste_data.csv
│
├── models/
│   ├── waste_decision_tree.pkl
│   ├── waste_random_forest.pkl
│   ├── waste_svm.pkl
│   ├── collection_decision_tree.pkl
│   ├── collection_random_forest.pkl
│   ├── collection_svm.pkl
│   ├── waste_scaler.pkl
│   ├── waste_encoder.pkl
│   ├── collection_scaler.pkl
│   └── collection_encoder.pkl
│
├── output/
│   └── model_accuracy.csv
│
├── graphs/
│   ├── waste_distribution.png
│   ├── collection_distribution.png
│   ├── waste_weight.png
│   ├── correlation_heatmap.png
│   ├── moisture_by_waste.png
│   ├── bin_fill_collection.png
│   └── model_accuracy_comparison.png
├── src/
│   ├── check_dataset.py
│   ├── data_exploration.py
│   ├── eval.py
│   ├──train_models.py 
│
├── preprocessing.py
├── model_comparison.py
├── main.py
├── README.md
└── requirements.txt
```

---

## ⚙️ 12. How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### Step 2: Run Dataset Checking

```bash
python check_dataset.py
```

### Step 3: Run Data Exploration

```bash
python data_exploration.py
```

### Step 4: Run Data Preprocessing

```bash
python preprocessing.py
```

### Step 5: Train Machine Learning Models

```bash
python train_models.py
```

### Step 6: Evaluate Models

```bash
python eval.py
```

### Step 7: Generate Model Comparison Graph

```bash
python model_comparison.py
```

### Step 8: Run Streamlit Application

```bash
streamlit run main.py
```

---

## ⚠️ 13. Project Limitations

- The dataset contains only 30 records.
- Model performance may change when tested on a larger dataset.
- Predictions depend on the quality and accuracy of input data.
- More real-world waste samples are needed for reliable deployment.

---

## 🚀 14. Future Improvements

- Increase the dataset size with real-world waste records.
- Improve model accuracy through additional data and testing.
- Integrate real-time smart dustbin sensors.
- Add image-based waste classification using Deep Learning.
- Develop a mobile application.
- Integrate GPS-based waste collection planning.
- Deploy the application online.

---

## 🏁 15. Conclusion

The **Intelligent Waste Segregation and Collection Prediction System** demonstrates how Machine Learning can support smart waste management.

The project combines waste classification, collection requirement prediction, model evaluation, and an interactive Streamlit application.

By automating these processes, the system provides a foundation for more efficient waste segregation and collection planning.

---

## 👩‍💻 16. Project Information

- **Project Title:** Intelligent Waste Segregation and Collection Prediction Using Machine Learning
- **Developer:** Arpita Thakur
- **Department:** B.Tech CSE (AI & ML)
- **Institution:** Ghani Khan Choudhury Institute of Engineering & Technology (GKCIET), Malda
- **Guided By:** Dr. Showmik Bhowmik
- **Project Guide LinkedIn:** [Dr. Showmik Bhowmik](https://www.linkedin.com/in/showmik-bhowmik-phd-b9380726/)

---

## 🔖 17. Keywords

`Machine Learning` `Waste Management` `Waste Classification` `Collection Prediction` `Decision Tree` `Random Forest` `SVM` `Python` `Streamlit` `Data Science`
