# SecureSense Web Application - Complete Setup Guide

## 📦 Prerequisites

Before running the web application, ensure you have:

- **Python 3.8+** installed (check with `python --version`)
- **pip** package manager (check with `pip --version`)
- **Git** (optional, for cloning)
- At least **500MB** free disk space
- A modern web browser (Chrome, Firefox, Edge, Safari)

## 🚀 First-Time Setup

### Step 1: Get the Code

**Option A: Clone from GitHub**
```bash
git clone https://github.com/khuynh22/SecureSense-A-Data-Driven-Framework-for-Phishing-Attack-Prevention.git
cd SecureSense-A-Data-Driven-Framework-for-Phishing-Attack-Prevention
```

**Option B: Download ZIP**
1. Download repository as ZIP from GitHub
2. Extract to your desired location
3. Open terminal/PowerShell in that directory

### Step 2: Create Virtual Environment

A virtual environment isolates project dependencies from your system Python.

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux (Bash/Zsh):**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

**Verify activation:** Your terminal prompt should show `(.venv)` at the beginning.

### Step 3: Install Python Packages

With the virtual environment activated:

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Installation will include:**
- Flask 3.0.0 (web framework)
- pandas 2.1.3 (data processing)
- scikit-learn 1.3.2 (ML algorithms)
- plotly 5.18.0 (visualizations)
- And other required packages

**Expected installation time:** 2-5 minutes depending on internet speed.

### Step 4: Verify Installation

Check that key packages are installed:

```bash
python -c "import flask, pandas, sklearn, plotly; print('✓ All packages installed successfully!')"
```

If you see the success message, you're ready to go!

## 🎯 Running the Web Application

### Starting the Server

1. **Ensure virtual environment is activated:**
   ```bash
   # Windows
   .venv\Scripts\Activate.ps1

   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Start Flask application:**
   ```bash
   python app.py
   ```

3. **Look for these messages:**
   ```
   * Serving Flask app 'app'
   * Debug mode: on
   * Running on http://127.0.0.1:5000
   * Restarting with stat
   * Debugger is active!
   ```

4. **Access the application:**
   - Open your browser
   - Go to: `http://127.0.0.1:5000` or `http://localhost:5000`
   - You should see the SecureSense upload interface

### Using the Application

1. **Upload Dataset:**
   - Click "Browse Files" or drag-and-drop your CSV file
   - **Required CSV format:**
     - Must have `CLASS_LABEL` or `labels` column
     - Values: `0` = Legitimate, `1` = Phishing
     - Should have 48 feature columns (URL, domain, content features)
   - Example: Use `Phishing_Legitimate_full.csv` included in the repo

2. **Analyze:**
   - Click "Analyze Dataset" button
   - Training progress shown in terminal
   - Wait 10-30 seconds (depends on dataset size)

3. **View Results:**
   - Automatically redirected to results page
   - See performance metrics for all 3 models:
     - Decision Tree (optimized)
     - Logistic Regression
     - Random Forest
   - Interactive Plotly visualizations:
     - Model comparison bar chart
     - Confusion matrices
     - Detailed metrics table

### Stopping the Server

Press `CTRL+C` in the terminal where Flask is running.

## 📂 Directory Structure

After setup, your project should look like:

```
SecureSense/
├── .venv/                          # Virtual environment (created)
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── Phishing_Legitimate_full.csv    # Sample dataset
├── templates/
│   ├── index.html                 # Upload page (front-end)
│   └── results.html               # Results page (front-end)
├── static/
│   └── css/
│       └── style.css              # Front-end styling
├── uploads/                        # Uploaded files (auto-created)
├── models/                         # Trained models (auto-created)
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
└── results.json                   # Latest analysis results
```

## 🎨 Front-End Details

### Technology Stack
- **HTML5** - Structure
- **CSS3** - Styling (see `static/css/style.css`)
- **JavaScript** (Vanilla) - File upload interaction
- **Plotly.js** - Interactive charts
- **Jinja2** - Flask templating engine

### Front-End Files

**`templates/index.html`** - Upload Interface
- Drag-and-drop file upload
- File validation (CSV only, max 16MB)
- AJAX form submission
- Responsive design

**`templates/results.html`** - Results Dashboard
- Dynamic data rendering via Jinja2
- Plotly interactive charts embedded
- Model metrics comparison table
- Confusion matrix heatmaps

**`static/css/style.css`** - Styling
- Modern gradient backgrounds
- Card-based layout
- Responsive grid system
- Hover effects and animations

### Customizing the Front-End

To modify the UI:

1. **Change colors/styles:** Edit `static/css/style.css`
2. **Modify layout:** Edit `templates/index.html` or `templates/results.html`
3. **Add visualizations:** Update `create_visualizations()` in `app.py`
4. **Refresh browser** to see changes (Flask auto-reloads in debug mode)

## ⚙️ Configuration

### Flask Settings (app.py)

```python
app.config['UPLOAD_FOLDER'] = 'uploads'           # Upload directory
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file
app.config['ALLOWED_EXTENSIONS'] = {'csv'}         # Only CSV files
```

**To change upload limit:** Modify `MAX_CONTENT_LENGTH` value (in bytes)

### Model Hyperparameters

These are research-validated values (do not change without revalidating):

```python
# Decision Tree - optimal pruning
DecisionTreeClassifier(ccp_alpha=0.010, random_state=42)

# Logistic Regression
LogisticRegression(max_iter=1000, random_state=42)

# Random Forest - balanced performance
RandomForestClassifier(n_estimators=100, max_depth=32, random_state=42)
```

### Port Configuration

Default port is **5000**. To change:
```python
# In app.py, line 265
app.run(debug=True, port=5001)  # Use different port
```

## 🔧 Troubleshooting

### Common Issues

**1. "python: command not found"**
- **Windows:** Use `python` or `py`
- **macOS/Linux:** Use `python3`
- Verify installation: `python --version`

**2. "No module named 'flask'"**
```bash
# Ensure virtual environment is activated
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

**3. "Address already in use" (Port 5000 busy)**
```bash
# Option A: Kill process using port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9

# Option B: Use different port (see Configuration)
```

**4. "Permission denied" when activating virtual environment (Windows)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**5. CSV upload fails: "must contain CLASS_LABEL or labels column"**
- Verify CSV has the correct target column name
- Check column contains binary values (0 or 1)
- Example:
  ```csv
  id,feature1,feature2,...,CLASS_LABEL
  1,0.5,0.3,...,0
  2,0.8,0.1,...,1
  ```

**6. Models train slowly**
- Dataset too large: Try with smaller sample first
- Reduce Random Forest estimators:
  ```python
  RandomForestClassifier(n_estimators=50, ...)  # Instead of 100
  ```

**7. Browser can't connect to localhost:5000**
- Check Flask is actually running (look for "Running on" message)
- Try `http://127.0.0.1:5000` instead of `localhost`
- Check firewall settings
- Ensure no VPN interfering with localhost

**8. Plotly charts not displaying**
- Check browser console for JavaScript errors (F12)
- Verify `plotly` package installed: `pip show plotly`
- Clear browser cache

### Getting Help

If issues persist:
1. Check terminal output for error messages
2. Look at browser console (F12 → Console tab)
3. Verify all dependencies installed: `pip list`
4. Review Flask logs in terminal
5. Open an issue on GitHub with:
   - Error messages
   - Python version
   - Operating system
   - Steps to reproduce

## 📊 Understanding Results

### Metrics Explained

**Accuracy**: Overall correctness (correct predictions / total predictions)
- 94%+ means excellent performance

**Precision**: Of predicted phishing sites, how many were actually phishing?
- High precision = Few false positives

**Recall**: Of actual phishing sites, how many were detected?
- High recall = Few false negatives (missed phishing sites)

**F1-Score**: Harmonic mean of precision and recall
- Balanced measure of model effectiveness

**Confusion Matrix**:
```
                Predicted
              Legit  Phish
Actual Legit  [TN]   [FP]
       Phish  [FN]   [TP]
```
- TN (True Negative): Correctly identified legitimate
- TP (True Positive): Correctly identified phishing
- FP (False Positive): Legitimate flagged as phishing
- FN (False Negative): Phishing missed (dangerous!)

## 🚀 Advanced Usage

### Running in Production

**Don't use Flask's built-in server for production!** Use Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t securesense .
docker run -p 5000:5000 securesense
```

### API Endpoint (Future Feature)

Current web app supports browser uploads only. For programmatic access:

```python
import requests

# Upload CSV via API
files = {'file': open('dataset.csv', 'rb')}
response = requests.post('http://localhost:5000/upload', files=files)
print(response.json())
```

## 📝 Features

- ✅ **Drag-and-drop upload** interface
- ✅ **Multi-model training** (3 algorithms simultaneously)
- ✅ **Interactive visualizations** with Plotly
- ✅ **Detailed metrics** (accuracy, precision, recall, F1)
- ✅ **Confusion matrices** for error analysis
- ✅ **Model persistence** (saved as .pkl files)
- ✅ **Responsive design** (works on mobile/tablet)
- ✅ **Real-time feedback** during training

## 📚 Additional Resources

- **Main README**: [`README.md`](README.md) - Project overview and research details
- **Dataset**: `Phishing_Legitimate_full.csv` - 10K balanced samples
- **Research Poster**: `Phishing Attacks Poster.pdf` - UIC Expo 2023
- **Notebooks**: `.ipynb` files for exploratory analysis

## 🎓 Learning Resources

Want to understand the code better?

- **Flask Docs**: https://flask.palletsprojects.com/
- **scikit-learn Tutorials**: https://scikit-learn.org/stable/tutorial/
- **Plotly Python**: https://plotly.com/python/
- **ML Fundamentals**: Review notebook files for detailed comments
