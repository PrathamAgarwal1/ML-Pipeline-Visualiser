from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Use non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'ml_pipeline_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASETS_FOLDER'] = 'datasets'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global-ish state (in a real app, use a DB or Redis, here we use session + local CSV)
DATA_FILE = 'processed_data.csv'

def get_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

@app.route('/')
def index():
    preloaded = [f for f in os.listdir(app.config['DATASETS_FOLDER']) if f.endswith('.csv')]
    return render_template('index.html', preloaded=preloaded)

@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    dataset_name = request.form.get('dataset')
    if dataset_name:
        path = os.path.join(app.config['DATASETS_FOLDER'], dataset_name)
        df = pd.read_csv(path)
        save_data(df)
        session['dataset_name'] = dataset_name
        session['target_column'] = None
        return redirect(url_for('preprocessing'))
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        filename = file.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        df = pd.read_csv(path)
        save_data(df)
        session['dataset_name'] = filename
        session['target_column'] = None
        return redirect(url_for('preprocessing'))
    return redirect(url_for('index'))

@app.route('/preprocessing')
def preprocessing():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    preview = df.head(10).to_html(classes='table table-striped table-hover')
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return render_template('preprocessing.html', preview=preview, info=info)

@app.route('/handle_missing', methods=['POST'])
def handle_missing():
    strategy = request.form.get('strategy') # mean, median, mode, drop
    df = get_data()
    if df is not None:
        for col in df.columns:
            if df[col].isnull().any():
                if strategy == 'mean' and df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median' and df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == 'drop':
                    df = df.dropna()
                    break
        save_data(df)
    return redirect(url_for('preprocessing'))

@app.route('/encode', methods=['POST'])
def encode():
    df = get_data()
    if df is not None:
        # Label Encoding for all objects for simplicity in this visualizer
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        save_data(df)
    return redirect(url_for('preprocessing'))

@app.route('/scale', methods=['POST'])
def scale():
    method = request.form.get('method') # standard, minmax
    df = get_data()
    if df is not None:
        # Scale all numeric columns except target if set
        cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        target = session.get('target_column')
        if target in cols_to_scale:
            cols_to_scale.remove(target)
            
        if method == 'standard':
            scaler = StandardScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        save_data(df)
    return redirect(url_for('preprocessing'))

@app.route('/analysis')
def analysis():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    
    stats = df.describe().to_html(classes='table table-sm table-bordered')
    corr = df.corr(numeric_only=True)
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    heatmap_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    columns = df.columns.tolist()
    return render_template('analysis.html', stats=stats, heatmap=heatmap_url, columns=columns)

@app.route('/get_plot', methods=['POST'])
def get_plot():
    plot_type = request.form.get('plot_type')
    col1 = request.form.get('col1')
    col2 = request.form.get('col2')
    
    df = get_data()
    plt.figure(figsize=(8, 5))
    
    if plot_type == 'histogram':
        sns.histplot(df[col1], kde=True)
        plt.title(f'Histogram of {col1}')
    elif plot_type == 'boxplot':
        sns.boxplot(y=df[col1])
        plt.title(f'Boxplot of {col1}')
    elif plot_type == 'scatter':
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f'Scatter Plot: {col1} vs {col2}')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({'plot_url': f'data:image/png;base64,{plot_url}'})

@app.route('/train_page')
def train_page():
    df = get_data()
    if df is None:
        return redirect(url_for('index'))
    columns = df.columns.tolist()
    return render_template('train.html', columns=columns)

@app.route('/train', methods=['POST'])
def train_model():
    problem_type = request.form.get('problem_type')
    target = request.form.get('target')
    model_name = request.form.get('model')
    session['target_column'] = target
    session['problem_type'] = problem_type
    
    df = get_data()
    if df is None: return redirect(url_for('index'))
    
    X = df.drop(columns=[target]) if target in df.columns else df
    # Filter to only numeric columns to prevent Skilllearn conversion errors
    X = X.select_dtypes(include=[np.number])
    
    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how='all')
    
    if X.empty:
        return "Error: No valid numeric features found (all features were empty or non-numeric)."
    
    # Impute NaNs with Mean, fallback to 0 if mean is NaN
    X = X.fillna(X.mean()).fillna(0)
    
    y = df[target] if target in df.columns else None
    
    # Handle NaNs in y (Drop rows)
    if y is not None:
        mask = y.notnull()
        X = X[mask]
        y = y[mask]
    
    if len(X) < 2:
        return "Error: Not enough data points (at least 2 required) after cleaning missing values."

    # Validate target for Regression
    if problem_type == 'regression' and df[target].dtype == 'object':
        return f"Error: Target column '{target}' contains text but is being used for Regression. Please select a numeric target or switch to Classification."

    # Final cast to float to be absolutely sure
    X = X.astype(float)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if y is not None else (X, X, None, None)
    except Exception as e:
        return f"Error during data split: {str(e)}"
    
    model = None
    results = {}
    
    try:
        if problem_type == 'classification':
            if model_name == 'logistic': model = LogisticRegression(max_iter=1000)
            elif model_name == 'knn': model = KNeighborsClassifier()
            elif model_name == 'svm': model = SVC()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['cm'] = confusion_matrix(y_test, y_pred).tolist()
            report = classification_report(y_test, y_pred, output_dict=True)
            results['report'] = {k.replace(' ', '_'): v for k, v in report.items()}
            
            # Advanced: Cross-val (adjust cv based on class counts)
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            cv_folds = min(5, min_class_count)
            if cv_folds >= 2:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds)
                results['cv_mean'] = cv_scores.mean()
            else:
                results['cv_mean'] = results['accuracy'] # Fallback

        elif problem_type == 'regression':
            if model_name == 'linear': model = LinearRegression()
            elif model_name == 'rf': model = RandomForestRegressor()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            results['r2'] = r2_score(y_test, y_pred)
            # Advanced: Cross-val
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2')
            results['cv_mean'] = cv_scores.mean()
            
            # Advanced: Feature Importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(X.columns, model.feature_importances_.tolist()))
                results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
        elif problem_type == 'clustering':
            if model_name == 'kmeans': model = KMeans(n_clusters=3, n_init=10)
            elif model_name == 'hierarchical': model = AgglomerativeClustering(n_clusters=3)
            
            clusters = model.fit_predict(X)
            results['silhouette'] = silhouette_score(X, clusters) if len(set(clusters)) > 1 else 0
    except Exception as e:
        return f"Error during model training: {str(e)}"
        
    # Save model and update session state atomically
    joblib.dump(model, 'models/last_model.joblib')
    session['results'] = results
    session['model_name'] = model_name
    session['features'] = X.columns.tolist()
    
    return redirect(url_for('evaluation'))

@app.route('/evaluation')
def evaluation():
    results = session.get('results')
    model_name = session.get('model_name')
    problem_type = session.get('problem_type')
    if not results:
        return redirect(url_for('train_page'))
    return render_template('evaluation.html', results=results, model_name=model_name, problem_type=problem_type)

@app.route('/prediction')
def prediction():
    features = session.get('features')
    if not features:
        return redirect(url_for('train_page'))
    return render_template('prediction.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('models/last_model.joblib')
    features = session.get('features')
    
    # Get values for trained features in correct order
    inputs = []
    for col in features:
        val = request.form.get(col)
        inputs.append(float(val) if val else 0.0)
    
    prediction = model.predict([inputs])[0]
    return render_template('prediction.html', features=features, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
