import nbformat as nbf

def create_notebook(title, cells_data, filename):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9"
        }
    }
    nb.cells = []
    
    # Add title
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}"))
    
    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))
            
    with open(filename, 'w') as f:
        nbf.write(nb, f)

# Notebook 1: Data Preprocessing
nb1_cells = [
    ("markdown", "This notebook contains the preprocessing functions used in the pipeline. It is imported directly by `app.py`."),
    ("code", "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder"),
    ("markdown", "### 1. Handle Missing Values Function"),
    ("code", "def handle_missing_values(df, strategy):\n    for col in df.columns:\n        if df[col].isnull().any():\n            if strategy == 'mean' and df[col].dtype in [np.float64, np.int64]:\n                df[col] = df[col].fillna(df[col].mean())\n            elif strategy == 'median' and df[col].dtype in [np.float64, np.int64]:\n                df[col] = df[col].fillna(df[col].median())\n            elif strategy == 'mode':\n                df[col] = df[col].fillna(df[col].mode()[0])\n            elif strategy == 'drop':\n                df = df.dropna()\n                break\n    return df"),
    ("markdown", "### 2. Categorical Encoding Function"),
    ("code", "def encode_categorical(df):\n    le = LabelEncoder()\n    for col in df.select_dtypes(include=['object']).columns:\n        df[col] = le.fit_transform(df[col].astype(str))\n    return df"),
    ("markdown", "### 3. Feature Scaling Function"),
    ("code", "def scale_features(df, method, target_column=None):\n    cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()\n    if target_column and target_column in cols_to_scale:\n        cols_to_scale.remove(target_column)\n        \n    if method == 'standard':\n        scaler = StandardScaler()\n        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])\n    elif method == 'minmax':\n        scaler = MinMaxScaler()\n        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])\n    return df")
]

# Notebook 2: Exploratory Data Analysis
nb2_cells = [
    ("markdown", "This notebook contains the Exploratory Data Analysis (EDA) functions used in the pipeline. It is imported directly by `app.py`."),
    ("code", "import numpy as np\nimport pandas as pd\nimport io\nimport base64\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport seaborn as sns"),
    ("markdown", "### 1. Descriptive Statistics"),
    ("code", "def generate_descriptive_stats(df):\n    return df.describe().to_html(classes='table table-sm table-bordered')"),
    ("markdown", "### 2. Correlation Heatmap"),
    ("code", "def generate_correlation_heatmap(df):\n    corr = df.corr(numeric_only=True)\n    plt.figure(figsize=(10, 8))\n    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=\".2f\")\n    img = io.BytesIO()\n    plt.savefig(img, format='png', bbox_inches='tight')\n    img.seek(0)\n    heatmap_url = base64.b64encode(img.getvalue()).decode()\n    plt.close()\n    return heatmap_url"),
    ("markdown", "### 3. Dynamic Plottings"),
    ("code", "def generate_plot(df, plot_type, col1, col2=None):\n    plt.figure(figsize=(8, 5))\n    if plot_type == 'histogram':\n        sns.histplot(df[col1], kde=True)\n        plt.title(f'Histogram of {col1}')\n    elif plot_type == 'boxplot':\n        sns.boxplot(y=df[col1])\n        plt.title(f'Boxplot of {col1}')\n    elif plot_type == 'scatter' and col2:\n        sns.scatterplot(x=df[col1], y=df[col2])\n        plt.title(f'Scatter Plot: {col1} vs {col2}')\n        \n    img = io.BytesIO()\n    plt.savefig(img, format='png', bbox_inches='tight')\n    img.seek(0)\n    plot_url = base64.b64encode(img.getvalue()).decode()\n    plt.close()\n    return f'data:image/png;base64,{plot_url}'")
]

# Notebook 3: Model Training
nb3_cells = [
    ("markdown", "This notebook demonstrates the models trained for Classification, Regression, and Clustering. It is imported directly by `app.py`."),
    ("code", "import numpy as np\nimport pandas as pd\nimport joblib\nimport os\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.linear_model import LogisticRegression, LinearRegression\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.cluster import KMeans, AgglomerativeClustering\nfrom sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, silhouette_score"),
    ("markdown", "### 1. Data Preparation for Models"),
    ("code", "def prepare_data_for_training(df, target, problem_type):\n    X = df.drop(columns=[target]) if target in df.columns else df.copy()\n    X = X.select_dtypes(include=[np.number])\n    X = X.dropna(axis=1, how='all')\n    \n    if X.empty:\n        raise ValueError(\"Error: No valid numeric features found.\")\n    X = X.fillna(X.mean()).fillna(0)\n    y = df[target] if target in df.columns else None\n    \n    if y is not None:\n        mask = y.notnull()\n        X = X[mask]\n        y = y[mask]\n    if len(X) < 2:\n        raise ValueError(\"Error: Not enough data points.\")\n    if problem_type == 'regression' and target in df.columns and df[target].dtype == 'object':\n        raise ValueError(f\"Error: Target '{target}' is text but being used for Regression.\")\n    if problem_type == 'classification' and y is not None:\n        if len(y.unique()) > 20 and pd.api.types.is_numeric_dtype(y):\n            raise ValueError(f\"Error: Target '{target}' has too many unique numerical values ({len(y.unique())}) for Classification. Try 'Regression' instead!\")\n        \n    X = X.astype(float)\n    return X, y"),
    ("markdown", "### 2. General Training and Evaluation Logic"),
    ("code", "def train_and_evaluate_model(X, y, problem_type, model_name, save_path='models/last_model.joblib'):\n    try:\n        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if y is not None else (X, X, None, None)\n    except Exception as e:\n        raise ValueError(f\"Error during data split: {str(e)}\")\n        \n    model = None\n    results = {}\n    \n    try:\n        if problem_type == 'classification':\n            if model_name == 'logistic': model = LogisticRegression(max_iter=1000)\n            elif model_name == 'knn': model = KNeighborsClassifier()\n            elif model_name == 'svm': model = SVC()\n            else: raise ValueError(\"Invalid classification model\")\n            \n            model.fit(X_train, y_train)\n            y_pred = model.predict(X_test)\n            results['accuracy'] = accuracy_score(y_test, y_pred)\n            results['cm'] = confusion_matrix(y_test, y_pred).tolist()\n            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)\n            results['report'] = {k.replace(' ', '_'): v for k, v in report.items()}\n            \n            class_counts = y.value_counts()\n            cv_folds = min(5, class_counts.min())\n            if cv_folds >= 2:\n                results['cv_mean'] = cross_val_score(model, X, y, cv=cv_folds).mean()\n            else:\n                results['cv_mean'] = results['accuracy']\n\n        elif problem_type == 'regression':\n            if model_name == 'linear': model = LinearRegression()\n            elif model_name == 'rf': model = RandomForestRegressor()\n            else: raise ValueError(\"Invalid regression model\")\n            \n            model.fit(X_train, y_train)\n            y_pred = model.predict(X_test)\n            results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))\n            results['r2'] = r2_score(y_test, y_pred)\n            results['cv_mean'] = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2').mean()\n            if hasattr(model, 'feature_importances_'):\n                importance = dict(zip(X.columns, model.feature_importances_.tolist()))\n                results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))\n                \n        elif problem_type == 'clustering':\n            if model_name == 'kmeans': model = KMeans(n_clusters=3, n_init=10)\n            elif model_name == 'hierarchical': model = AgglomerativeClustering(n_clusters=3)\n            else: raise ValueError(\"Invalid clustering model\")\n            \n            clusters = model.fit_predict(X)\n            results['silhouette'] = silhouette_score(X, clusters) if len(set(clusters)) > 1 else 0\n            \n    except Exception as e:\n        raise ValueError(f\"Error during model training: {str(e)}\")\n        \n    os.makedirs(os.path.dirname(save_path), exist_ok=True)\n    joblib.dump(model, save_path)\n    return model, results")
]

create_notebook("Data Preprocessing Step-by-Step", nb1_cells, "data_preprocessing.ipynb")
create_notebook("Exploratory Data Analysis", nb2_cells, "analysis.ipynb")
create_notebook("Model Training and Evaluation", nb3_cells, "model_training.ipynb")
print("Successfully generated notebooks as module formats.")
