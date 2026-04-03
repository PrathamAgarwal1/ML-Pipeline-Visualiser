import streamlit as st
import pandas as pd
import numpy as np

# Import natively from Jupyter Notebooks using ipynb hook
import ipynb.fs.defs.data_preprocessing as data_preprocessing
import ipynb.fs.defs.analysis as analysis
import ipynb.fs.defs.model_training as model_training

st.set_page_config(page_title="ML Pipeline Visualiser", layout="wide")

st.title("ML Pipeline Visualiser")
st.markdown("This project is a highly explainable, ML interface. All backend logic is natively queried from Jupyter Notebooks.")

# Upload Dataset
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.header("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())
    
    # Preprocessing Module
    st.sidebar.header("2. Preprocessing")
    missing_strategy = st.sidebar.selectbox("Missing Values Strategy", ["mean", "median", "mode", "drop"])
    if st.sidebar.button("Handle Missing Values"):
        df = data_preprocessing.handle_missing_values(df, strategy=missing_strategy)
        st.success("Missing Values Handled!")
        
    if st.sidebar.button("Auto-Encode Categorical Features"):
        df = data_preprocessing.encode_categorical(df)
        st.success("Encoded Categorical Strings to Numeric!")
        
    scale_method = st.sidebar.selectbox("Scale Features", ["None", "standard", "minmax"])
    # Determine potential target variable so we don't scale it
    target_col = st.sidebar.selectbox("Target Variable (ignore for scaling)", ["None"] + df.columns.tolist())
    
    if st.sidebar.button("Scale Numerical Features"):
        if scale_method != "None":
            t_col = None if target_col == "None" else target_col
            df = data_preprocessing.scale_features(df, method=scale_method, target_column=t_col)
            st.success(f"Features Scaled using {scale_method.capitalize()} Scaler!")

    with st.expander("View Preprocessed Data"):
        st.dataframe(df.head())

    # Analysis Module (EDA)
    st.header("Exploratory Data Analysis")
    if st.checkbox("Show Descriptive Statistics"):
        # The notebook function returns HTML, we can render it safely in Streamlit
        stats_html = analysis.generate_descriptive_stats(df)
        st.markdown(stats_html, unsafe_allow_html=True)
        
    if st.checkbox("Show Correlation Heatmap"):
        heatmap_b64 = analysis.generate_correlation_heatmap(df)
        st.markdown(f'<img src="data:image/png;base64,{heatmap_b64}">', unsafe_allow_html=True)

    # Model Training Module
    st.sidebar.header("3. Model Training")
    problem_type = st.sidebar.selectbox("Problem Type", ["classification", "regression", "clustering"])
    
    if problem_type == "classification":
        model_name = st.sidebar.selectbox("Model", ["logistic", "knn", "svm"])
    elif problem_type == "regression":
        model_name = st.sidebar.selectbox("Model", ["linear", "rf"])
    else:
        model_name = st.sidebar.selectbox("Model", ["kmeans", "hierarchical"])
        
    if st.sidebar.button("Train Model"):
        st.header("Training Results")
        try:
            if target_col == "None" and problem_type != "clustering":
                st.error("Please select a valid Target Variable from the Preprocessing section!")
            else:
                # Prepare Data
                t_col = None if target_col == "None" else target_col
                X, y = model_training.prepare_data_for_training(df, target=t_col, problem_type=problem_type)
                
                # Train
                st.info(f"Training {model_name} model for {problem_type}...")
                model, results = model_training.train_and_evaluate_model(X, y, problem_type, model_name)
                
                # Use the new explainability evaluation module
                import model_evaluation
                model_evaluation.display_model_evaluation(
                    model, 
                    model_name, 
                    problem_type, 
                    X, y, 
                    results, 
                    t_col, 
                    scale_method
                )
                    
                st.success("Model trained and safely saved to models/last_model.joblib!")
                
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
else:
    st.info("Awaiting dataset upload. Please use the sidebar to begin.")
