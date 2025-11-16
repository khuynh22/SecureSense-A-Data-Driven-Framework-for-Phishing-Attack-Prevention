# SecureSense Copilot Instructions

## Project Overview
Machine learning phishing detection framework with Flask web interface. Three models (Decision Tree, Logistic Regression, Random Forest) analyze 48 website features to classify phishing vs legitimate sites with 94%+ accuracy.

## Architecture

### Components
- **Web App** (`app.py`): Flask backend handling file uploads, model training, and results visualization
- **ML Scripts**: Standalone Python scripts for each algorithm (originally Colab notebooks)
- **Notebooks**: Exploratory analysis with visualization (`.ipynb` files)
- **Dataset**: `Phishing_Legitimate_full.csv` (10k balanced samples, 48 features + `CLASS_LABEL` or `labels` column)

### Data Flow
1. User uploads CSV via `/upload` endpoint
2. `train_models()` validates data, splits 80/20, trains all 3 models
3. Models saved to `models/` directory as `.pkl` files
4. Results + Plotly visualizations saved to `results.json`
5. User redirected to `/results` showing interactive metrics

## Critical Conventions

### Model Hyperparameters (DO NOT CHANGE)
These are research-validated values from the capstone project:

```python
# Decision Tree - optimized via cost-complexity pruning analysis
DecisionTreeClassifier(ccp_alpha=0.010, random_state=42)

# Logistic Regression
LogisticRegression(max_iter=1000, random_state=42)

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=32, random_state=42)
```

**Why:** `ccp_alpha=0.010` prevents overfitting in Decision Tree (see pruning analysis in `decision_tree_for_phishing_attack.py` lines 118-142). Random Forest params balance accuracy vs computational cost.

### Data Splitting
Always use `train_test_split(test_size=0.2, random_state=42)` for reproducibility across all models.

### Feature Engineering Pattern
```python
# app.py follows this pattern for all models:
if 'CLASS_LABEL' in data.columns:
    data = data.rename(columns={'CLASS_LABEL': 'labels'})

X = data.drop(['labels'], axis=1)
if 'id' in X.columns:
    X = X.drop(['id'], axis=1)  # Remove ID before training
y = data['labels']
```

**Important:** Dataset may have `CLASS_LABEL` or `labels` - always normalize to `labels`. The `id` column (if present) must be excluded from features.

### Mutual Information for Feature Selection
Use `sklearn.feature_selection.mutual_info_classif` (not correlation) for feature ranking - captures non-linear relationships better than Spearman. See `phishing_detection_using_logistic_regression_and_random_forest_classifier.py` lines 155-162.

## Development Workflows

### Running the Web App
```powershell
# Activate virtual environment first (if exists)
.venv\Scripts\Activate.ps1

# Run Flask app
python app.py
# Opens at http://127.0.0.1:5000
```

**Debug Mode:** App runs with `debug=True` for development - change for production deployment.

### Testing Models
Upload `Phishing_Legitimate_full.csv` through web interface or run standalone scripts:
```powershell
python decision_tree_for_phishing_attack.py
python phishing_detection_using_logistic_regression_and_random_forest_classifier.py
```

### Working with Notebooks
Notebooks are exploratory - they fetch data from GitHub raw URL, not local file:
```python
# Pattern used in all notebooks
ml = pd.read_csv('https://raw.githubusercontent.com/khuynh22/Phishing-Detection/main/Phishing_Legitimate_full.csv')
```

## Code Style

### Linting
- **flake8** with `max-line-length = 100` (see `.flake8`)
- Follow this for all Python files

### Documentation Style
Scripts use triple-quoted docstrings explaining research methodology:
```python
"""## Prune Tree
Removing impurities data from the dataset to find valuable branches for tree
pruning. Tree pruning reduces the model's size, enhances its functionality,
and prevents overfitting.
"""
```

## Key Files Reference

### Web Application
- `app.py`: Complete Flask app with model training pipeline
- `templates/index.html`: Upload interface with drag-drop
- `templates/results.html`: Results visualization page
- `static/css/style.css`: Styling for web UI

### ML Implementation
- `decision_tree_for_phishing_attack.py`: Decision tree with pruning analysis
- `phishing_detection_using_logistic_regression_and_random_forest_classifier.py`: Comparative analysis using mutual information

### Research Artifacts
- `README.md`: Full project documentation with performance metrics
- `WEB_APP_GUIDE.md`: Quick start guide for web interface
- `Phishing Attacks Poster.pdf`: Research poster (UIC Expo 2023)

## Common Tasks

### Adding a New Model
1. Add training logic in `train_models()` function in `app.py`
2. Follow existing pattern: fit, predict, compute metrics
3. Add confusion matrix to `create_visualizations()`
4. Update `results.html` template to display new model

### Modifying Features
48-feature dataset is fixed for this project. To change:
1. Update dataset with new columns
2. Retrain all models (they auto-detect features)
3. Update README.md feature descriptions if adding new categories

### Deployment Considerations
- Remove `debug=True` from `app.run()`
- Use production WSGI server (Gunicorn recommended)
- Set `MAX_CONTENT_LENGTH` appropriately for your use case
- Create `uploads/` and `models/` directories on server
