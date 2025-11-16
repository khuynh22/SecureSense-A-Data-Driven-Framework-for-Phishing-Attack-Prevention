# SecureSense Web Application

## Quick Start Guide

### Running the Web Application

1. **Start the server:**
   ```bash
   .venv/bin/python app.py
   ```

2. **Open your browser:**
   Navigate to: `http://127.0.0.1:5000`

3. **Upload your dataset:**
   - Click "Browse Files" or drag and drop your CSV file
   - The CSV must contain a `CLASS_LABEL` or `labels` column with values 0 (legitimate) and 1 (phishing)
   - Click "Analyze Dataset" to train models and see results

### Features

- **Upload Interface**: Drag-and-drop or browse to upload CSV files
- **Multi-Model Training**: Automatically trains Decision Tree, Logistic Regression, and Random Forest models
- **Interactive Visualizations**: View performance metrics with interactive Plotly charts
- **Detailed Results**: See accuracy, precision, recall, F1-score, and confusion matrices for each model

### File Structure

```
SecureSense/
├── app.py                  # Flask application
├── templates/
│   ├── index.html         # Upload page
│   └── results.html       # Results page
├── static/
│   └── css/
│       └── style.css      # Styling
├── uploads/               # Uploaded files (auto-created)
├── models/                # Saved models (auto-created)
└── results.json          # Analysis results (auto-generated)
```

### Usage Example

1. Upload `Phishing_Legitimate_full.csv` (included in the repo)
2. Wait for models to train (usually 10-30 seconds)
3. View comprehensive results including:
   - Model performance comparison
   - Confusion matrices
   - Detailed metrics for each model
   - Key insights

### Notes

- Maximum file size: 16MB
- Supported format: CSV only
- The app runs in debug mode for development
- For production deployment, use a WSGI server like Gunicorn

### Stopping the Server

Press `CTRL+C` in the terminal where the server is running.
