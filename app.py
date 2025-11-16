"""
SecureSense Web Application
A Flask-based web interface for phishing detection using machine learning
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def train_models(data):
    """Train all three models and return them with performance metrics"""
    # Prepare data
    if 'CLASS_LABEL' in data.columns:
        data.rename(columns={'CLASS_LABEL': 'labels'}, inplace=True)

    # Separate features and labels
    X = data.drop(['labels'], axis=1)
    if 'id' in X.columns:
        X = X.drop(['id'], axis=1)
    y = data['labels']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    results = {}

    # 1. Decision Tree
    dt_model = DecisionTreeClassifier(ccp_alpha=0.010, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    models['decision_tree'] = dt_model
    results['decision_tree'] = {
        'name': 'Decision Tree',
        'accuracy': accuracy_score(y_test, dt_pred),
        'precision': precision_score(y_test, dt_pred),
        'recall': recall_score(y_test, dt_pred),
        'f1_score': f1_score(y_test, dt_pred),
        'confusion_matrix': confusion_matrix(y_test, dt_pred).tolist(),
        'predictions': dt_pred.tolist()
    }

    # 2. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'name': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'f1_score': f1_score(y_test, lr_pred),
        'confusion_matrix': confusion_matrix(y_test, lr_pred).tolist(),
        'predictions': lr_pred.tolist()
    }

    # 3. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    models['random_forest'] = rf_model
    results['random_forest'] = {
        'name': 'Random Forest',
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred),
        'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist(),
        'predictions': rf_pred.tolist()
    }

    # Store test data info
    results['test_data'] = {
        'y_test': y_test.tolist(),
        'total_samples': len(y_test),
        'phishing_samples': int(sum(y_test == 1)),
        'legitimate_samples': int(sum(y_test == 0))
    }

    return models, results

def create_visualizations(results):
    """Create interactive visualizations using Plotly"""
    plots = {}

    # 1. Model Comparison Bar Chart
    model_names = [results[m]['name'] for m in ['decision_tree', 'logistic_regression', 'random_forest']]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    fig_comparison = go.Figure()
    for metric in metrics:
        values = [results[m][metric] for m in ['decision_tree', 'logistic_regression', 'random_forest']]
        fig_comparison.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=model_names,
            y=values,
            text=[f'{v:.4f}' for v in values],
            textposition='auto',
        ))

    fig_comparison.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    plots['comparison'] = json.dumps(fig_comparison, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Confusion Matrices for each model
    for model_key in ['decision_tree', 'logistic_regression', 'random_forest']:
        cm = results[model_key]['confusion_matrix']

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Legitimate', 'Predicted Phishing'],
            y=['Actual Legitimate', 'Actual Phishing'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))

        fig_cm.update_layout(
            title=f'Confusion Matrix - {results[model_key]["name"]}',
            height=400
        )
        plots[f'cm_{model_key}'] = json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder)

    return plots

@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and model training"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and process data
            data = pd.read_csv(filepath)

            # Validate data
            if 'CLASS_LABEL' not in data.columns and 'labels' not in data.columns:
                return jsonify({'error': 'CSV must contain a "CLASS_LABEL" or "labels" column'}), 400

            # Train models
            models, results = train_models(data)

            # Create visualizations
            plots = create_visualizations(results)

            # Save models
            for model_name, model in models.items():
                joblib.dump(model, f'models/{model_name}.pkl')

            # Save results for display
            with open('results.json', 'w') as f:
                json.dump({'results': results, 'plots': plots}, f)

            return jsonify({
                'success': True,
                'message': 'Models trained successfully!',
                'redirect': url_for('results')
            })

        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/results')
def results():
    """Display results page"""
    try:
        with open('results.json', 'r') as f:
            data = json.load(f)
        return render_template('results.html',
                             results=data['results'],
                             plots=data['plots'])
    except FileNotFoundError:
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)
