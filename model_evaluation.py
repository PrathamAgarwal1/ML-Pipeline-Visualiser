import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

MODEL_INFO = {
    'linear': {
        'name': 'Linear Regression', 'type': 'Regression',
        'desc': 'Predicts a continuous value by fitting a straight line that minimizes the sum of squared errors between actual and predicted values.'
    },
    'logistic': {
        'name': 'Logistic Regression', 'type': 'Classification',
        'desc': 'Predicts categorical class probabilities using a sigmoid function, mapping any real value to a probability between 0 and 1.'
    },
    'knn': {
        'name': 'K-Nearest Neighbors (KNN)', 'type': 'Classification',
        'desc': 'Predicts the class of a data point based on the majority voting of its "k" closest neighbors in the feature space.'
    },
    'svm': {
        'name': 'Support Vector Machine (SVM)', 'type': 'Classification',
        'desc': 'Finds the optimal hyperplane that maximizes the margin (distance) between different classes in the feature space.'
    },
    'rf': {
        'name': 'Random Forest', 'type': 'Regression',
        'desc': 'An ensemble method that builds multiple decision trees based on random subsets of data and averages their outputs to prevent overfitting.'
    },
    'kmeans': {
        'name': 'K-Means Clustering', 'type': 'Clustering',
        'desc': 'Partitions dataset into k distinct, non-overlapping clusters based on distance to the centroid (mean) of each cluster.'
    },
    'hierarchical': {
        'name': 'Hierarchical Clustering', 'type': 'Clustering',
        'desc': 'Builds a hierarchy of clusters using a bottom-up (agglomerative) approach, pairing data points based on proximity.'
    }
}

def display_model_evaluation(model, model_name, problem_type, X, y, results, target_col, scale_method):
    st.markdown("---")
    
    info = MODEL_INFO.get(model_name, {'name': model_name.capitalize(), 'type': problem_type.capitalize(), 'desc': 'A machine learning model.'})

    st.header(f"Model Evaluation: {info['name']}")
    
    # 1. MODEL OVERVIEW
    st.subheader("1. MODEL OVERVIEW")
    st.write(f"**Name of the model**: {info['name']}")
    st.write(f"**Type**: {info['type']}")
    st.info(f"**How it works**: {info['desc']}")
    
    # 2. INPUT & OUTPUT
    st.subheader("2. INPUT & OUTPUT")
    X_cols = list(X.columns)
    st.write(f"**Input features (X)**: {', '.join(X_cols)}")
    
    if problem_type in ['classification', 'regression'] and target_col:
        st.write(f"**Target variable (Y)**: {target_col}")
        
    out_type_map = {'classification': 'Class Label (Discrete)', 'regression': 'Continuous Value', 'clustering': 'Cluster Assignment'}
    st.write(f"**Type of output**: {out_type_map.get(problem_type, 'Unknown')}")
    
    # 3. MODEL WORKING
    st.subheader("3. MODEL WORKING")
    if problem_type in ['regression', 'classification']:
        if 'feature_importance' in results:
            st.write("**Feature Importance / Parameters:**")
            imp_df = pd.DataFrame(list(results['feature_importance'].items()), columns=['Feature', 'Value'])
            st.dataframe(imp_df.head(10))
        else:
            st.write("Model does not expose direct feature importances or coefficients.")
    elif problem_type == 'clustering':
        unique_clusters = len(np.unique(results.get('y_pred', [])))
        st.write(f"**Number of clusters formed**: {unique_clusters}")
        if 'cluster_centers' in results:
            st.write("**Centroids (KMeans):**")
            st.dataframe(pd.DataFrame(results['cluster_centers'], columns=X.columns).head())
        if model_name == 'hierarchical':
            st.write("**Dendrogram Data Structure Generated** (See Visualization block).")

    # 4. PREDICTIONS
    st.subheader("4. PREDICTIONS")
    if problem_type in ['regression', 'classification']:
        X_test = results.get('X_test')
        y_test = results.get('y_test')
        y_pred = results.get('y_pred')
        
        if X_test is not None and y_test is not None and y_pred is not None:
            pred_df = pd.DataFrame({'Actual': np.array(y_test)[:5], 'Predicted': np.array(y_pred)[:5]})
            if problem_type == 'regression':
                pred_df['Error Difference'] = np.abs(pred_df['Actual'] - pred_df['Predicted'])
                def highlight_error(val):
                    color = 'red' if val > pred_df['Error Difference'].mean() else 'green'
                    return f'color: {color}'
                st.dataframe(pred_df.style.applymap(highlight_error, subset=['Error Difference']))
            else:
                pred_df['Match'] = pred_df['Actual'] == pred_df['Predicted']
                def highlight_match(val):
                    color = 'green' if val else 'red'
                    return f'color: {color}'
                st.dataframe(pred_df.style.applymap(highlight_match, subset=['Match']))
        else:
            st.write("Prediction data could not be retrieved.")
    elif problem_type == 'clustering':
        cl_df = X.copy().head()
        cl_df['Cluster'] = results.get('y_pred', [])[:5]
        st.write("**First 5 rows with cluster assignments:**")
        st.dataframe(cl_df)

    # 5. EVALUATION METRICS
    st.subheader("5. EVALUATION METRICS")
    if problem_type == 'regression':
        st.metric(label="RMSE", value=f"{results.get('rmse', 0):.4f}")
        st.metric(label="R² Score", value=f"{results.get('r2', 0):.4f}")
    elif problem_type == 'classification':
        st.metric(label="Accuracy", value=f"{results.get('accuracy', 0):.4f}")
        cm = results.get('cm')
        if cm:
            st.write("**Confusion Matrix Definition**:")
            st.write(f"TN: {cm[0][0]} | FP: {cm[0][1] if len(cm[0])>1 else 0}")
            if len(cm) > 1:
                st.write(f"FN: {cm[1][0]} | TP: {cm[1][1] if len(cm[1])>1 else 0}")
            st.dataframe(pd.DataFrame(cm))
    elif problem_type == 'clustering':
        st.metric(label="Silhouette Score", value=f"{results.get('silhouette', 0):.4f}")

    # 6. VISUALIZATION
    st.subheader("6. VISUALIZATION")
    try:
        if problem_type == 'regression':
            y_test = results.get('y_test')
            y_pred = results.get('y_pred')
            if y_test is not None and y_pred is not None:
                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                fig.add_shape(type="line", x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test), line=dict(dash="dash", color="red"))
                st.plotly_chart(fig, use_container_width=True)
                
        elif problem_type == 'classification':
            cm = results.get('cm')
            if cm:
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix Heatmap",
                                labels=dict(x="Predicted Label", y="True Label", color="Count"))
                st.plotly_chart(fig, use_container_width=True)
                
        elif problem_type == 'clustering':
            preds = results.get('y_pred')
            if preds is not None:
                # Use PCA for 2D visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=[str(c) for c in preds], title="Cluster Scatter Plot (PCA 2D)", labels={'color': 'Cluster'})
                st.plotly_chart(fig, use_container_width=True)
                
            if model_name == 'hierarchical':
                st.write("**Dendrogram:**")
                fig, ax = plt.subplots(figsize=(10, 5))
                # Subsample for speed if dataset is huge, max 100
                Z = linkage(X.head(100), 'ward')
                dendrogram(Z, ax=ax)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render visualizations: {str(e)}")

    # 7. INTERPRETATION
    st.subheader("7. INTERPRETATION")
    if problem_type == 'regression':
        r2 = results.get('r2', 0)
        if r2 < 0:
            st.warning("Model performance is **poor**. R² is negative, meaning model is worse than a simple baseline (mean prediction).")
        elif r2 < 0.5:
            st.info("Model performance is **moderate**. There is significant unexplained variance.")
        else:
            st.success("Model performance is **good**. It captures the underlying trend well.")
            
    elif problem_type == 'classification':
        acc = results.get('accuracy', 0)
        if acc < 0.5:
            st.warning("Model performance is **poor**. Accuracy is extremely low.")
        elif acc < 0.75:
            st.info("Model performance is **moderate**.")
        else:
            st.success("Model performance is **good**.")
            
    elif problem_type == 'clustering':
        sil = results.get('silhouette', 0)
        if sil < 0.2:
            st.warning("Clustering is **poor** or highly overlapping.")
        elif sil < 0.5:
            st.info("Clusters are **passable** but might not be completely distinct.")
        else:
            st.success("Clusters are **well separted** and defined.")

    # 8. DEBUG INSIGHTS
    st.subheader("8. DEBUG INSIGHTS")
    warnings_found = False
    
    if problem_type == 'regression' and results.get('r2', 0) < 0:
        st.error("⚠️ Model is completely failing to learn (R² < 0). Check if relationship is non-linear or data is too noisy.")
        warnings_found = True
        
    if problem_type == 'classification' and results.get('accuracy', 0) < 0.5:
        st.error("⚠️ Accuracy is worse than random guessing or data is heavily imbalanced.")
        warnings_found = True
        
    if model_name in ['knn', 'svm', 'kmeans', 'hierarchical'] and scale_method in ['None', None]:
        st.error(f"⚠️ **Distance-based model ({info['name']}) used but data is NOT scaled!** Scaling is highly recommended (e.g. Standard/MinMax).")
        warnings_found = True

    if target_col is not None and problem_type == 'classification' and y.dtype.kind in 'fc':
        st.warning("⚠️ Target variable appears to be continuous but parsed for classification. Ensure classes are properly encoded.")
        warnings_found = True
        
    if not warnings_found:
        st.success("No immediate debug warnings found. Processing looks healthy!")

    if 'feature_importance' in results and len(results['feature_importance']) > 0:
        st.write("**Feature Importance Ranking:**")
        imp_df = pd.DataFrame(list(results['feature_importance'].items())[:10], columns=['Feature', 'Value'])
        fig = px.bar(imp_df, x='Value', y='Feature', orientation='h', title="Top 10 Feature Importance")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig)
        
    if problem_type == 'classification' and model_name in ['knn', 'svm', 'logistic']:
        st.write("**2D Decision Boundary (PCA projection):**")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        if len(X_pca) > 0 and len(np.unique(y)) > 1:
            try:
                # Train a temporary model just for boundary plotting
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                
                if model_name == 'knn': plot_model = KNeighborsClassifier()
                elif model_name == 'svm': plot_model = SVC(kernel='linear')
                else: plot_model = LogisticRegression()
                
                plot_model.fit(X_pca, y)

                x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                fig = go.Figure()
                fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.1), y=np.arange(y_min, y_max, 0.1), z=Z, colorscale='viridis', opacity=0.3, showscale=False))
                fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers', marker=dict(color=y, colorscale='viridis', line=dict(width=1))))
                fig.update_layout(title="Decision Boundary (2D PCA projection)", xaxis_title="PCA 1", yaxis_title="PCA 2")
                st.plotly_chart(fig)
            except Exception as e:
                st.write("Could not plot decision boundary.")
