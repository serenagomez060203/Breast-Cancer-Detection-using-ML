# Breast Cancer Detection using Machine Learning

A comprehensive machine learning project for classifying breast cancer diagnoses (Benign vs. Malignant) using the classic Wisconsin Breast Cancer Dataset. This project demonstrates a complete end-to-end workflow from data preprocessing to model deployment and evaluation.

##  Project Overview

This project aims to build a robust binary classifier to predict breast cancer diagnosis based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The models achieved high accuracy in distinguishing between malignant (M) and benign (B) tumors.

##  Tech Stack

*   **Programming Language:** Python
*   **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
*   **ML Models:** Logistic Regression, Random Forest Classifier
*   **Techniques:** Data Standardization, PCA, K-Fold Cross-Validation, Hyperparameter Tuning (GridSearchCV)

##  Dataset

*   **Source:** Wisconsin Breast Cancer Diagnostic dataset (included as `b cancer dataset.csv`).
*   **Features:** 30 numerical features (like `radius_mean`, `texture_mean`, `area_mean`, etc.) computed from cell nuclei images.
*   **Target Variable:** `diagnosis` (M = Malignant, B = Benign).

##  Steps

1.  **Data Loading & Cleaning:** Removed irrelevant columns (`id`, `Unnamed: 32`) and encoded the target variable.
2.  **Exploratory Data Analysis (EDA):** Conducted univariate and bivariate analysis, including correlation heatmaps.
3.  **Data Preprocessing:** Standardized features using `StandardScaler`.
4.  **Dimensionality Reduction:** Applied Principal Component Analysis (PCA) to reduce feature space while retaining ~95% variance.
5.  **Model Building & Evaluation:**
    *   Trained and evaluated **Logistic Regression** and **Random Forest** models.
    *   Utilized **K-Fold Cross-Validation** for robust performance estimation.
    *   Performed **Hyperparameter Tuning** using `GridSearchCV`.
6.  **Visualization:** Plotted decision boundaries, confusion matrices, and learning curves.

## 📈 Results

The tuned Logistic Regression model achieved the following performance metrics on the test set:
*   **Accuracy:** ~97%
*   **Precision:** ~97%
*   **Recall:** ~96%
*   **F1-Score:** ~96%

## ▶️ How to Run in Google Colab

1.  **Open Google Colab:** Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
2.  **Copy the Code:** Copy the entire Python code block from the `Breast Cancer Detection using PCA.ipynb` file (or the section below) into a code cell in your Colab notebook.
3.  **Upload the Dataset:**
    *   Run the code cell. It will prompt you to upload the `b cancer dataset.csv` file.
    *   **Important:** When the upload prompt appears, select the `b cancer dataset.csv` file from your computer.
4.  **Run All:** Execute the cell. The code will run the entire analysis and display all outputs, including graphs and results.

---

*For a detailed walkthrough of the code and analysis, please refer to the Jupyter Notebook file (`Breast Cancer Detection using PCA.ipynb`) in this repository.*
