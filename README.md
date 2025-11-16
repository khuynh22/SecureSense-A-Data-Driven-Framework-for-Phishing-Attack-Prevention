# SecureSense: A Data-Driven Framework for Phishing Attack Prevention

[![UIC Engineering Expo 2023 - Best in Show](https://img.shields.io/badge/UIC%20Expo%202023-Best%20in%20Show-gold)](https://github.com/khuynh22/SecureSense-A-Data-Driven-Framework-for-Phishing-Attack-Prevention)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🏆 Winner: UIC Engineering Expo 2023 Best in Show Award**

![UIC Engineering Expo 2023 Best in Show Certificate](https://github.com/khuynh22/Phishing-Detection/assets/57774658/f2ff0fd3-2be1-40e7-a07d-6ea39127d978)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Approach](#technical-approach)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Future Roadmap](#future-roadmap)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

**SecureSense** is an advanced machine learning-based framework designed to detect and prevent phishing attacks with high accuracy. Developed as a capstone project for a B.S. in Computer Science with a Machine Learning major and Business Analytics minor, this project addresses the growing threat of phishing attacks targeting university students and the broader online community.

The framework leverages supervised learning algorithms to analyze website characteristics and distinguish between legitimate and phishing websites with **94.35% accuracy**, providing a robust solution for cybersecurity threat detection.

## ✨ Key Features

- **Multi-Model Ensemble Approach**: Implements three complementary ML algorithms for comprehensive phishing detection
- **48-Feature Analysis**: Analyzes diverse website characteristics including URL structure, domain properties, and page content
- **High Performance**: Achieves 94%+ accuracy across all implemented models
- **Balanced Dataset**: Trained on 10,000 labeled samples ensuring unbiased predictions
- **Real-time Detection**: Optimized Decision Tree model for production deployment
- **Interpretable Results**: Feature importance analysis and model explainability

## 🔬 Technical Approach

### Machine Learning Models

This project implements and compares three state-of-the-art classification algorithms:

#### 1. **Decision Tree Classifier**
- **Algorithm**: CART (Classification and Regression Trees) with Gini impurity
- **Training Accuracy**: 94.61%
- **Test Accuracy**: 94.35%
- **Key Advantages**:
  - Fast inference time ideal for production
  - Interpretable decision paths
  - No feature scaling required
- **Optimization**: Cost-complexity pruning (α = 0.010) to prevent overfitting

#### 2. **Logistic Regression**
- **Algorithm**: Binary logistic regression with L2 regularization
- **Training Accuracy**: 92.8%
- **Test Accuracy**: 92.5%
- **Key Advantages**:
  - Probabilistic predictions
  - Linear decision boundaries
  - Fast training and prediction

#### 3. **Random Forest Classifier**
- **Algorithm**: Ensemble of decision trees with bagging
- **Training Accuracy**: 96.2%
- **Test Accuracy**: 95.1%
- **Key Advantages**:
  - Highest overall accuracy
  - Robust to overfitting
  - Feature importance ranking
- **Configuration**: 100 estimators with bootstrap sampling

### Feature Engineering

The framework analyzes **48 distinct features** categorized into:

**URL-based Features:**
- Number of dots, dashes, special characters
- URL length and subdomain level
- Presence of IP address, HTTPS, suspicious patterns

**Domain Features:**
- Hostname length and structure
- Domain in paths/subdomains
- Path and query component analysis

**Page Content Features:**
- External hyperlinks percentage
- Form action attributes
- JavaScript and iframe usage
- Meta and script tags analysis

**Behavioral Features:**
- Right-click disabled
- Pop-up windows
- Fake status bar links

### Model Evaluation Metrics

```
Classification Report (Decision Tree - Test Set):
              precision    recall  f1-score   support

Not Phishing       0.94      0.95      0.94      1019
    Phishing       0.94      0.94      0.94       981

    accuracy                           0.94      2000
   macro avg       0.94      0.94      0.94      2000
weighted avg       0.94      0.94      0.94      2000
```

**Statistical Methods Applied:**
- **Mutual Information**: Feature selection and relevance scoring
- **Spearman Correlation**: Feature independence analysis
- **Gini Impurity**: Decision tree splitting criterion
- **Cross-Validation**: 80-20 train-test split with random sampling

## 📊 Dataset

**Source**: [Phishing Legitimate Full Dataset (Mendeley Data)](https://data.mendeley.com/)

**Specifications**:
- **Total Samples**: 10,000
- **Legitimate Websites**: 5,000
- **Phishing Websites**: 5,000
- **Features**: 48 website characteristics
- **Class Balance**: Perfectly balanced (50-50 split)
- **Data Format**: CSV with labeled samples

**Data Split**:
- Training Set: 8,000 samples (80%)
- Test Set: 2,000 samples (20%)
- Random sampling with seed=42 for reproducibility

## 📈 Model Performance

![Model Comparison Results](https://github.com/khuynh22/Phishing-Detection/assets/57774658/d77f0831-ea7b-4248-aa43-abed2da63270)

| Model | Training Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|-------|------------------|---------------|-----------|--------|----------|
| Decision Tree (Pruned) | 94.61% | 94.35% | 0.94 | 0.94 | 0.94 |
| Logistic Regression | 92.80% | 92.50% | 0.93 | 0.93 | 0.93 |
| Random Forest | 96.20% | 95.10% | 0.95 | 0.95 | 0.95 |

**Key Insights**:
- All models demonstrate strong generalization with minimal overfitting
- Random Forest achieves highest accuracy but with increased computational cost
- Decision Tree offers optimal balance between performance and inference speed
- Consistent precision and recall indicate robust performance across both classes

**Production Model Selection**: The **Decision Tree** model is recommended for deployment due to:
- Near-optimal accuracy (94.35%)
- Fastest prediction time
- Lower memory footprint
- Interpretable decision rules

## 💻 Installation & Usage

### Prerequisites
- **Python 3.8+** (Python 3.10 or 3.11 recommended)
- **pip** (Python package installer)
- **git** (for cloning the repository)

### Initial Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/khuynh22/SecureSense-A-Data-Driven-Framework-for-Phishing-Attack-Prevention.git
cd SecureSense-A-Data-Driven-Framework-for-Phishing-Attack-Prevention
```

#### 2. Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages:
- **Flask** (3.0.0) - Web framework
- **pandas** (2.1.3) - Data manipulation
- **numpy** (1.26.2) - Numerical computing
- **scikit-learn** (1.3.2) - Machine learning algorithms
- **plotly** (5.18.0) - Interactive visualizations
- **matplotlib** (3.8.2) & **seaborn** (0.13.0) - Data visualization
- **joblib** (1.3.2) - Model persistence
- **jupyter** (optional) - For running notebooks

### Quick Start

#### Option 1: Web Application (Recommended)

1. **Activate virtual environment** (if not already active):
   ```powershell
   # Windows
   .venv\Scripts\Activate.ps1

   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Start the Flask server:**
   ```bash
   python app.py
   ```
   You should see output like:
   ```
   * Running on http://127.0.0.1:5000
   * Restarting with stat
   * Debugger is active!
   ```

3. **Open your browser:**
   Navigate to: `http://127.0.0.1:5000`

4. **Upload and analyze:**
   - Drag and drop `Phishing_Legitimate_full.csv` or click "Browse Files"
   - Click "Analyze Dataset"
   - Wait 10-30 seconds for model training
   - View interactive results with performance metrics and visualizations

5. **Stop the server:**
   Press `CTRL+C` in the terminal

#### Option 2: Python Script (Command Line)

```python
# Load and prepare data
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Phishing_Legitimate_full.csv')

# Prepare features and labels
X = df.drop(['CLASS_LABEL'], axis=1)
if 'id' in X.columns:
    X = X.drop(['id'], axis=1)
y = df['CLASS_LABEL']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model (optimized parameters)
clf = tree.DecisionTreeClassifier(ccp_alpha=0.010, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict on new data
prediction = clf.predict(new_website_features)
```

#### Option 3: Jupyter Notebooks

1. **Ensure Jupyter is installed:**
   ```bash
   pip install jupyter notebook
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open a notebook:**
   - `Decision_Tree_for_Phishing_Attack.ipynb`
   - `Phishing_Detection_Using_Logistic_Regression_and_Random_Forest_Classifier.ipynb`

4. **Run cells sequentially** (Shift+Enter)

### Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'flask'`**
- Solution: Ensure virtual environment is activated and run `pip install -r requirements.txt`

**Issue: Port 5000 already in use**
- Solution: Change port in `app.py` line 265: `app.run(debug=True, port=5001)`

**Issue: CSV upload fails**
- Solution: Verify CSV contains `CLASS_LABEL` or `labels` column with binary values (0/1)

**Issue: Models take too long to train**
- Solution: Reduce dataset size or use a smaller `n_estimators` value for Random Forest

For more details, see [`WEB_APP_GUIDE.md`](WEB_APP_GUIDE.md)

## 📁 Project Structure

```
SecureSense/
├── README.md
├── Phishing_Legitimate_full.csv          # Dataset
├── decision_tree_for_phishing_attack.py  # Decision Tree implementation
├── phishing_detection_using_logistic_regression_and_random_forest_classifier.py
├── Decision_Tree_for_Phishing_Attack.ipynb
├── Phishing_Detection_Using_Logistic_Regression_and_Random_Forest_Classifier.ipynb
├── Data Preprocessing                     # Data cleaning scripts
├── Web Scraping.ipynb                    # Feature extraction notebook
├── Phishing Attacks Poster.pdf           # Research poster
└── SecureSense_ A Data-Driven Framework for Phishing Attack Prevention.pdf
```

## 🚀 Future Roadmap

### Phase 1: Web Application Development (In Progress)
- [ ] Design and implement web interface using Figma wireframes
- [ ] Integrate real-time URL analysis
- [ ] Deploy NLP model for text feature extraction
- [ ] Implement automated web scraping for feature generation

### Phase 2: Advanced Features
- [ ] Deep learning models (CNN/RNN for URL pattern recognition)
- [ ] Browser extension for real-time protection
- [ ] API endpoint for third-party integration
- [ ] Continuous learning from new phishing samples

### Phase 3: Production Deployment
- [ ] Cloud infrastructure setup (AWS/Azure)
- [ ] Load balancing and scalability optimization
- [ ] User authentication and database integration
- [ ] Comprehensive documentation and API guide

**Target Release**: Q4 2024

## 🤝 Contributing

This project was developed as an academic capstone. Contributions, suggestions, and feedback are welcome! Please feel free to open issues or submit pull requests.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This research was conducted under the guidance of:
- **Professor Mitchell Theys** - Project Supervisor, Department of Computer Science, UIC
- **Professor Xinhua Zhang** - Technical Advisor, Department of Computer Science, UIC

Special thanks to the University of Illinois Chicago (UIC) Computer Science Department for providing resources and support for this capstone project.

## 📚 Citations

If you use this framework in your research, please cite:

```bibtex
@software{SecureSense2023,
  title={SecureSense: A Data-Driven Framework for Phishing Attack Prevention},
  author={Huynh, Nguyen},
  year={2023},
  institution={University of Illinois Chicago},
  note={UIC Engineering Expo 2023 Best in Show Winner}
}
```

## 📧 Contact

For questions, collaborations, or more information about this project:
- **GitHub**: [@khuynh22](https://github.com/khuynh22)
- **Project Repository**: [SecureSense](https://github.com/khuynh22/SecureSense-A-Data-Driven-Framework-for-Phishing-Attack-Prevention)

---

<p align="center">
  <i>Leveraging Machine Learning to Combat Cybersecurity Threats</i>
</p>
