# Minimalist ML Pipeline Visualiser

A sleek, lightweight, and highly explainable Machine Learning Pipeline built for academic demonstrations. This project drops heavy web-framework boilerplate in favor of a minimalist **Streamlit** dashboard natively powered by **Jupyter Notebooks**, making the entire ML lifecycle—from preprocessing to model evaluation—fully transparent and easy to explain.

## 🎯 Project Overview

Building an ML model involves complex steps like cleaning data, handling missing values, scaling, and picking the right algorithms. This **ML Pipeline Visualiser** provides an intuitive interface to perform these tasks interactively.

What sets this project apart is its **Explainability Architecture**: the backend logic doesn't live in obfuscated scripts. Instead, the Streamlit app imports logic directly out of documented Jupyter Notebooks (`.ipynb`). You can open the notebooks to read the theory, understand the exact code executing behind the scenes, and then see the visual results in the app!

## 🚀 Features

*   **Native Notebook Flow**: Logic is natively sourced from `data_preprocessing.ipynb`, `analysis.ipynb`, and `model_training.ipynb`.
*   **Dataset Upload**: Dynamically ingest custom `.csv` data pipelines.
*   **Interactive Preprocessing**: Handle missing values (mean/median/mode/drop), auto-encode categorical strings, and scale numerical features (Standard/MinMax).
*   **EDA & Visuals**: View dynamically generated correlation heatmaps and descriptive statistics.
*   **Multi-Model Training**: Train and compare Classification (Logistic, KNN, SVM), Regression (Linear, Random Forest), or unsupervised Clustering (KMeans, Hierarchical).
*   **Performance Metrics**: Real-time stats like Accuracy, RMSE, Silhouette Scores, and Confusion matrices.

## 🛠️ Tech Stack

*   **Frontend / UI**: Streamlit
*   **Machine Learning**: Scikit-learn, Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Module Engine**: `ipynb` (Jupyter Notebook hook)

## 📋 How to Run Locally

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/PrathamAgarwal1/ML-Pipeline-Visualiser.git
    cd ML-Pipeline-Visualiser
    ```

2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn ipynb
    ```

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

4.  **View the App**:
    Your browser will automatically open the dashboard at `http://localhost:8501`.

## 📂 Project Structure

```text
ML-Pipeline-Visualiser/
├── app.py                      # Minimalist Streamlit Dashboard
├── data_preprocessing.ipynb    # Explainable Preprocessing module
├── analysis.ipynb              # Explainable EDA module
├── model_training.ipynb        # Explainable Training/Evaluation module
├── system_documentation.md     # Platform architecture docs
├── datasets/                   # Test CSV files
└── models/                     # Saved .joblib model dumps
```

## 🧠 Learning Outcomes

Building this project focused on:
*   Bridging the gap between raw Python ML execution and minimalist UI tools like Streamlit.
*   Understanding the importance of robust preprocessing logic (NaN handling, string decoding).
*   How to build transparent, "glass-box" architectures where the executing code doubles as the academic documentation (via Jupyter bindings).

---
