# 🏥 Smart Disease Predictor

An end-to-end Machine Learning web application for disease prediction from symptoms.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models (one-time setup)
```bash
python train_model.py
```

### 3. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure
```
smart_disease_predictor/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Saved models & metadata
│   ├── rf_model.pkl        # Random Forest model
│   ├── svm_model.pkl       # SVM model
│   ├── label_encoder.pkl   # Label encoder
│   ├── metadata.json       # Feature names & accuracies
│   └── disease_info.json   # Disease descriptions & precautions
└── data/                   # Dataset folder
    ├── training_data.csv
    └── test_data.csv
```

---

## 🧠 ML Models Used

| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest | ~97.6% | Ensemble of 200 decision trees |
| SVM (RBF kernel) | ~100% | Best generalization on this dataset |
| **Ensemble** | **Best** | Averages both model probabilities |

---

## 🩺 Features
- **41 diseases** covered
- **132 symptom** input features
- Real-time prediction with **confidence scores**
- Top-5 disease possibilities with probability bars
- Disease severity classification (Mild / Moderate / Serious / Critical)
- Precaution recommendations for each disease
- Disease explorer with search & filter
- Model insights dashboard with feature importance

---

## ⚠️ Disclaimer
This application is for **educational and research purposes only**.
It is NOT a substitute for professional medical diagnosis or advice.
Always consult a qualified healthcare professional for medical concerns.

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **ML Models**: scikit-learn (Random Forest, SVM)
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## 👤 Developer
Built as an end-to-end ML project for Smart Disease Prediction.
Dataset: 4920 samples, 41 diseases, 132 symptom features.
