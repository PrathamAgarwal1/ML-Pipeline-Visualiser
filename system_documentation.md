# Minimalist ML Pipeline Visualizer Documentation

This application provides a highly explainable, minimal Machine Learning pipeline dashboard using **Streamlit**. It strips away complex web layers like HTML and CSS, placing 100% of the focus directly on Machine Learning execution blocks. 

It is designed to be easily extensible, modular, and academically explainable for University presentations.

---

## 1. Architecture Overview

### Streamlit UI (`app.py`)
`app.py` is the entire User Interface! Powered by Streamlit, it behaves as a single native Python script that auto-generates dynamic dataframes, sidebar dropdowns, and data uploads with essentially zero HTML boilerplate. 

All routing and session logic has been removed. State flow is entirely top-to-bottom, matching exactly how a researcher would naturally execute an ML pipeline textually.

### Pipeline Modules (`ipynb` hooks)
The heavy-lifting logic is factored out into pure Jupyter Notebook files which act natively as modules. `app.py` uses the `ipynb.fs.defs` Python hook to import algorithms directly out of the notebooks! This completely bridges the gap between what is executing on the server and what is visually explainable in the lab:
- **`data_preprocessing.ipynb`**: Contains dataset operations like dropping/filling null values (`handle_missing_values`), encoding strings using Scikit-Learn LabelEncoding (`encode_categorical`), and mathematical feature normalization (`scale_features`).
- **`analysis.ipynb`**: Manages the visualizations. Uses Matplotlib in non-interactive `Agg` mode to output image buffer objects which are serialized into Base64 formats for safe network transmission.
- **`model_training.ipynb`**: Validates structural anomalies in training matrices (X) and vectors (y), fits dynamically-selected classifiers, regressors, or clustering algorithms using `scikit-learn`, logs performance metrics (like R2, Mean Squared Error, Accuracy, and Confusion Matrices) into a dictionary, and checkpoints model weights to local storage as binary dumps (`models/last_model.joblib`).

---

## 2. Explainability
Because `app.py` directly depends on these Notebooks, you can simply open any of the three `.ipynb` files to read the actual engine's logic accompanied by full Markdown textual explanations. There is absolutely no separation between the code you're trying to explain in presentation and what actually executes in the interface backend.

To launch the app:
1. Open terminal and type: `streamlit run app.py`
2. Test out your models in your browser.

---

## 3. Usage Requirements
- `streamlit`
- `ipynb`
- `scikit-learn`
- `pandas` & `numpy`
- `seaborn` & `matplotlib`
- `joblib` 
