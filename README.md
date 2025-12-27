# Diabetes Prediction System - Academic Project

> AI-powered diabetes prediction system using machine learning with ~90% accuracy.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This academic research project implements a machine learning-based diabetes prediction system using a pre-trained XGBoost classifier. The system analyzes medical diagnostic measurements to predict diabetes risk with high accuracy.

**Key Features:**

- ✅ Pre-trained XGBoost model with ~90% accuracy
- ✅ Interactive Streamlit web interface
- ✅ Real-time risk assessment
- ✅ Input validation and error handling
- ✅ Comprehensive medical disclaimers

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**

   ```bash
   streamlit run diabetes_prediction_app.py
   ```

4. **Access the application:**
   - The app will automatically open in your browser
   - Or navigate to: `http://localhost:8501`

## 💻 Usage

### Running the Application

**Single Command:**

```bash
streamlit run diabetes_prediction_app.py
```

### Using the Interface

1. **Enter Medical Parameters:**

   - Fill in all 8 input fields in the sidebar:
     - Pregnancies
     - Glucose Level
     - Blood Pressure
     - Skin Thickness
     - Insulin
     - BMI
     - Diabetes Pedigree Function
     - Age

2. **Run Prediction:**

   - Click "⚡ RUN PREDICTION ENGINE" button

3. **View Results:**
   - Risk level (Low / Moderate / High)
   - Diabetes probability percentage
   - Prediction confidence
   - System recommendations

## 📊 Model Information

### Algorithm

- **Model:** XGBoost (Gradient Boosting Classifier)
- **Accuracy:** 89.47% (Cross-Validation)
- **ROC-AUC:** 95.45%
- **Preprocessing:** RobustScaler for feature normalization
- **Feature Engineering:** Categorical features from BMI, Insulin, and Glucose

### Dataset

- **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Dataset:** Pima Indians Diabetes Database
- **Samples:** 768
- **Features:** 8 medical predictor variables

## 📁 Project Structure

```
Diabetes-Prediction/
│
├── diabetes_prediction_app.py    # Main Streamlit application (ONLY entry point)
├── model.pkl                      # Pre-trained XGBoost model
├── scaler.pkl                     # Pre-trained feature scaler
├── diabetes.csv                   # Dataset (for reference)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # License file
```

## 🔧 Technical Details

### Programming Concepts Demonstrated

1. **Object-Oriented Programming (OOP)**

   - `DiabetesPredictionApp` class with encapsulation
   - Class attributes and methods
   - Constructor initialization

2. **Functions**

   - Input validation function
   - Feature preparation function
   - Prediction function
   - Risk level calculation

3. **Data Structures**

   - Dictionary: Validation ranges for medical parameters
   - Lists: Feature ordering and categorical features
   - NumPy Arrays: Feature matrices for model input

4. **Machine Learning Integration**

   - Model loading using pickle
   - Feature scaling using pre-trained scaler
   - Probability prediction using predict_proba
   - Risk assessment based on probabilities

5. **Error Handling**
   - Try-except blocks for model loading
   - Input validation with user-friendly messages
   - Graceful error handling throughout

## ⚠️ Important Medical Disclaimer

**This application is developed as an academic research project for educational and research purposes only.**

### Medical Disclaimer

- This tool is **NOT a substitute** for professional medical advice, diagnosis, or treatment
- Always consult qualified healthcare professionals for medical decisions
- Do not rely solely on this prediction for health-related decisions
- The model accuracy is approximately 90%, not 100%
- Regular medical checkups are essential for proper health management

### Research Limitations

- **Dataset Limitation:** Model trained on Pima Indian heritage females (age ≥ 21 years)
- **Generalization:** Results may not generalize to all populations or demographics
- **Sample Size:** Training dataset consists of 768 samples
- **Feature Scope:** Limited to 8 diagnostic measurements

### Ethical Considerations

This research tool should be used responsibly and in conjunction with professional medical judgment. The developers and researchers are not liable for any medical decisions made based on this prediction system.

## 📝 Requirements

```
streamlit>=1.28.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## 🎓 Academic Use

This project demonstrates:

- Machine learning model deployment
- Interactive web application development with Streamlit
- Object-oriented programming principles
- Data preprocessing and feature engineering
- Model evaluation and prediction

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- Model: XGBoost (Gradient Boosting Classifier)

---

**Built for academic research and education purposes**
