# ML Pipeline Visualizer

A full-stack, interactive platform designed to simplify and visualize the end-to-end Machine Learning lifecycle. This project was built to demonstrate how data moves from raw CSV format through preprocessing, analysis, and training, finally resulting in a predictive model.

## 🎯 Project Overview

Building an ML model is more than just running a script; it involves a series of steps like cleaning data, handling missing values, and picking the right evaluation metrics. This **ML Pipeline Visualizer** provides a user-friendly dashboard to perform these tasks without writing code, making it an excellent tool for understanding the "behind-the-scenes" of scikit-learn.

Whether it's predicting Titanic survival (Classification), house prices (Regression), or segmenting customers (Clustering), this app handles it all in one flow.

## 🚀 Features

*   **Dataset Upload**: Support for custom CSV uploads or choosing from built-in datasets.
*   **Data Preprocessing**: Handle missing values (mean/median/mode/drop), label encoding for categorical data, and feature scaling (Standard/MinMax).
*   **Data Analysis**: Generate statistical summaries and dynamic correlation heatmaps.
*   **Model Training**: Train multiple models like Logistic Regression, KNN, Random Forest, and KMeans.
*   **Evaluation**: View performance metrics like Accuracy, RMSE, Confusion Matrices, and Feature Importance charts.
*   **Cross-Validation**: Automatic k-fold validation to ensure model reliability.
*   **Prediction Module**: Interactive form to input new data and get real-time predictions from the trained model.

## 🛠️ Tech Stack

*   **Backend**: Python (Flask)
*   **Machine Learning**: Scikit-learn, Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn, Chart.js
*   **Frontend**: HTML5, CSS3 (Custom Glassmorphism design), Bootstrap 5

## 📊 Datasets Included

1.  **Titanic Dataset**: Used for binary classification (Predicting survival).
2.  **House Prices Dataset**: Used for regression (Predicting property value).
3.  **Mall Customer Dataset**: Used for unsupervised clustering (Segmenting users based on spending habits).

Using multiple datasets helped in testing the robustness of the pipeline against different problem types and data distributions.

## 📋 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ml-pipeline-visualizer.git
    cd ml-pipeline-visualizer
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```

4.  **Open in Browser**:
    Go to `http://127.0.0.1:5000`

## 📂 Project Structure

```text
ML-Pipeline-Visualizer/
├── app.py              # Flask backend & ML logic
├── processed_data.csv  # Temporary storage for pipeline state
├── datasets/           # Preloaded CSV files
├── models/             # Saved .joblib model files
├── static/
│   └── css/style.css   # Custom dashboard styling
└── templates/          # Modular HTML components
```

## 📸 Screenshots
*(Add screenshots of your dashboard, training results, and evaluation pages here to showcase the UI)*

## 🧠 Learning Outcomes

Building this project helped me understand:
*   How to bridge the gap between Python ML scripts and a web-based UI.
*   The importance of robust preprocessing—handling NaNs and non-numeric data is where 80% of the work happens.
*   Interpreting model results through visualizations like heatmaps and importance bars.
*   Managing application state in Flask while working with large dataframes.

## 🏗️ Future Improvements

*   Adding more advanced models like XGBoost or LightGBM.
*   Implementing hyperparameter tuning (GridSearch/RandomSearch) via the UI.
*   Cloud deployment on platforms like Heroku or AWS.
*   Support for real-time API-based data ingestion.

---

**Author:** Shubhang Shrivastav  
**Course:** B.Tech CSE  
**Subject:** Machine Learning Lab
