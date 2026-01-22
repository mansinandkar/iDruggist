# iDruggist


## 🧩 Project Context
Cardiovascular diseases remain one of the leading causes of mortality worldwide. Accurate and early risk assessment using patient health indicators can significantly support clinical decision-making.

This project explores the use of **supervised machine learning models** to estimate the likelihood of a heart attack based on structured patient health data collected from multiple clinical sources. The emphasis is on **data integration, feature refinement, normalization, and comparative model evaluation**, rather than single-model optimization.

---

## 🏥 Use Case
**Goal:**  
Assist healthcare practitioners and analysts by providing a data-driven model that estimates heart attack risk using routinely available clinical measurements.

**Prediction Type:**  
Binary classification  
- `0` → Lower risk  
- `1` → Higher risk  

---

## 🧪 Data Sources & Composition
- **Dataset:** Heart Disease Dataset (UCI ML Repository)
- **Clinical Sources:** 4 independent clinics
- **Initial Feature Space:** 75+ clinical attributes
- **Final Feature Set:** 14 medically relevant attributes + target label

### Examples of Clinical Attributes
- Age, sex
- Chest pain type
- Resting blood pressure
- Cholesterol levels
- ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Number of major vessels
- Thalassemia category

---

## 🛠️ Data Engineering & Preparation

### Feature Reduction
- Reduced dimensionality from 75+ attributes to **14 clinically significant features**
- Eliminated noisy and redundant signals

### Data Integration
- Merged processed datasets from all clinics into a unified analytical dataset
- Integrated oxygen saturation statistics

### Data Cleaning
- No missing values in Cleveland subset
- Missing values in merged datasets handled using **mean imputation**

### Normalization
- Applied **StandardScaler** to normalize feature ranges
- Improved model stability and convergence

---

## 🧠 Modeling Strategy
Instead of tuning a single model, this project focuses on **algorithm comparison** to understand how different classifiers perform on the same clinical feature space.

### Models Evaluated
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

All models were trained using the same preprocessing pipeline and evaluated on a held-out test set.

---

## 📊 Model Performance Comparison

| Model | Accuracy (%) |
|------|--------------|
| Decision Tree | 77.86 |
| Logistic Regression | 85.19 |
| KNN | 86.74 |
| Random Forest | 86.67 |
| **Support Vector Machine (SVM)** | **87.67** |

---

## 🏆 Key Outcome
- **Support Vector Machine (SVM)** achieved the highest accuracy (87.67%)
- Demonstrated strong generalization on unseen clinical data
- Indicates suitability for structured, medium-dimensional healthcare datasets

---

## 🧰 Technology Stack

### Core Tools
- Python
- Jupyter Notebook / Google Colab

### Data Handling
- pandas
- NumPy

### Machine Learning
- scikit-learn
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- KNN

### Preprocessing
- StandardScaler
- Train/Test Split

---

## 🧠 Key Features
This project demonstrates:
- Practical handling of **multi-source clinical data**
- Feature reduction grounded in **domain relevance**
- Systematic **model benchmarking**
- Clear separation between data preparation and modeling logic

---

## 🔮 Extensions & Next Steps
- Introduce cross-validation and hyperparameter tuning
- Evaluate models using ROC-AUC and F1-score
- Add feature importance and explainability (SHAP)
- Convert the model into a REST-based clinical scoring service

---
