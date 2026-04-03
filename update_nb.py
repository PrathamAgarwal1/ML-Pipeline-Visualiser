import json

with open("d:/Programming/ML-Pipeline-Visualizer/model_training.ipynb", "r") as f:
    nb = json.load(f)

# Find the cell that contains train_and_evaluate_model
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any("def train_and_evaluate_model" in line for line in source):
            # We will replace the entire cell source
            new_source = """def train_and_evaluate_model(X, y, problem_type, model_name, save_path='models/last_model.joblib'):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if y is not None else (X, X, None, None)
    except Exception as e:
        raise ValueError(f"Error during data split: {str(e)}")
        
    model = None
    results = {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    try:
        if problem_type == 'classification':
            if model_name == 'logistic': model = LogisticRegression(max_iter=1000)
            elif model_name == 'knn': model = KNeighborsClassifier()
            elif model_name == 'svm': model = SVC()
            else: raise ValueError("Invalid classification model")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results['y_pred'] = y_pred
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['cm'] = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results['report'] = {k.replace(' ', '_'): v for k, v in report.items()}
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(X.columns, model.feature_importances_.tolist()))
                results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(X.columns, model.coef_[0].tolist()))
                results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
                
            class_counts = y.value_counts()
            cv_folds = min(5, class_counts.min())
            if cv_folds >= 2:
                results['cv_mean'] = cross_val_score(model, X, y, cv=cv_folds).mean()
            else:
                results['cv_mean'] = results['accuracy']

        elif problem_type == 'regression':
            if model_name == 'linear': model = LinearRegression()
            elif model_name == 'rf': model = RandomForestRegressor()
            else: raise ValueError("Invalid regression model")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results['y_pred'] = y_pred
            results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            results['r2'] = r2_score(y_test, y_pred)
            results['cv_mean'] = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2').mean()
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(X.columns, model.feature_importances_.tolist()))
                results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(X.columns, model.coef_.tolist()))
                results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
                
        elif problem_type == 'clustering':
            if model_name == 'kmeans': model = KMeans(n_clusters=3, n_init=10)
            elif model_name == 'hierarchical': model = AgglomerativeClustering(n_clusters=3)
            else: raise ValueError("Invalid clustering model")
            
            clusters = model.fit_predict(X)
            results['y_pred'] = clusters
            results['silhouette'] = silhouette_score(X, clusters) if len(set(clusters)) > 1 else 0
            if hasattr(model, 'cluster_centers_'):
                results['cluster_centers'] = model.cluster_centers_.tolist()
            
    except Exception as e:
        raise ValueError(f"Error during model training: {str(e)}")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model, results
"""
            # Need to format into list of lines with \n except last line
            lines = [line + '\\n' for line in new_source.split('\\n')]
            lines[-1] = lines[-1][:-2] # remote trailing \n literal that we just appended
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
            break

with open("d:/Programming/ML-Pipeline-Visualizer/model_training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated.")
