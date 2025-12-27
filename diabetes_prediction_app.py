"""
================================================================================
DIABETES PREDICTION SYSTEM - ACADEMIC PROJECT
================================================================================

PROJECT OVERVIEW:
-----------------
This program implements a machine learning-based diabetes prediction system
that uses a pre-trained XGBoost (Gradient Boosting) classifier to predict
whether a patient has diabetes based on medical diagnostic measurements.

DATASET INFORMATION:
--------------------
- Source: National Institute of Diabetes and Digestive and Kidney Diseases
- Dataset: Pima Indians Diabetes Database
- Features: 8 medical predictor variables
  * Pregnancies: Number of times pregnant
  * Glucose: Plasma glucose concentration (mg/dL)
  * BloodPressure: Diastolic blood pressure (mm Hg)
  * SkinThickness: Triceps skin fold thickness (mm)
  * Insulin: 2-Hour serum insulin (μU/ml)
  * BMI: Body mass index (kg/m²)
  * DiabetesPedigreeFunction: Diabetes pedigree function
  * Age: Age in years
- Target: Binary classification (0 = No Diabetes, 1 = Diabetes)
- Model Performance: ~90% accuracy with cross-validation

MODEL INFORMATION:
------------------
- Algorithm: XGBoost (Gradient Boosting Classifier)
- Preprocessing: RobustScaler for feature normalization
- Feature Engineering: Categorical features derived from BMI, Insulin, Glucose
- Model File: model.pkl (pre-trained)
- Scaler File: scaler.pkl (pre-trained)

WHY STREAMLIT WAS CHOSEN:
-------------------------
Streamlit was selected for this project because:
1. Rapid Development: Enables quick creation of interactive web applications
   without HTML/CSS/JavaScript knowledge
2. Python-Native: Pure Python implementation aligns with ML workflow
3. User-Friendly: Simple, intuitive interface for medical professionals
4. Academic Suitability: Demonstrates modern Python web application development
5. Deployment Ready: Easy to deploy and share with stakeholders

PROGRAMMING CONCEPTS DEMONSTRATED:
----------------------------------
1. Object-Oriented Programming (OOP)
   - DiabetesPredictionApp class with attributes and methods
   - Encapsulation of model loading and prediction logic
   - Class-based organization for maintainability

2. Functions
   - Input validation function
   - Feature preparation function
   - Prediction function
   - Risk level calculation function

3. Data Structures
   - Dictionary: Validation ranges for input constraints
   - Lists: Feature names, categorical feature creation
   - NumPy arrays: Feature arrays for model input

4. Machine Learning Integration
   - Model loading using pickle/joblib
   - Feature scaling using pre-trained scaler
   - Probability prediction using predict_proba
   - Risk assessment based on prediction probabilities

5. Error Handling
   - Try-except blocks for model loading
   - Input validation with user-friendly error messages
   - Graceful error handling throughout the application

================================================================================
"""

import streamlit as st
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional


# ============================================================================
# DATA STRUCTURES
# ============================================================================

# Dictionary: Medical validation ranges for input features
# This data structure stores the acceptable ranges for each medical parameter
# based on medical knowledge and dataset statistics
VALIDATION_RANGES: Dict[str, Tuple[float, float]] = {
    'Pregnancies': (0, 20),
    'Glucose': (0, 300),
    'BloodPressure': (0, 150),
    'SkinThickness': (0, 100),
    'Insulin': (0, 1000),
    'BMI': (0, 60),
    'DiabetesPedigreeFunction': (0, 3),
    'Age': (0, 120)
}

# List: Feature names in the order expected by the model
# This list ensures features are passed to the model in the correct sequence
FEATURE_ORDER: List[str] = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


# ============================================================================
# OBJECT-ORIENTED PROGRAMMING: MAIN CLASS
# ============================================================================

class DiabetesPredictionApp:
    """
    Main application class for diabetes prediction system.
    
    This class demonstrates Object-Oriented Programming (OOP) principles:
    - Encapsulation: Model and scaler are private class attributes
    - Methods: Organized functionality as class methods
    - Constructor: Initializes the application and loads models
    
    Attributes:
        model: Pre-trained XGBoost classifier (loaded from model.pkl)
        scaler: Pre-trained RobustScaler (loaded from scaler.pkl)
        numerical_cols: List of numerical feature column names
        categorical_cols: List of categorical feature column names
        model_loaded: Boolean indicating if model loaded successfully
    """
    
    def __init__(self, model_path: str = 'model.pkl', scaler_path: str = 'scaler.pkl'):
        """
        Constructor: Initializes the DiabetesPredictionApp class.
        
        This method demonstrates OOP initialization:
        - Sets up class attributes
        - Loads pre-trained model and scaler
        - Handles errors during model loading
        
        Args:
            model_path: Path to the pre-trained model file (model.pkl)
            scaler_path: Path to the pre-trained scaler file (scaler.pkl)
        """
        # Class attributes initialization
        self.model = None
        self.scaler = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.model_loaded = False
        
        # Error handling: Try to load model and scaler
        # Try multiple possible locations for model files
        possible_paths = [
            (model_path, scaler_path),  # Root directory
            ('flask/model.pkl', 'flask/scaler.pkl'),  # Flask directory
            (os.path.join(os.path.dirname(__file__), 'model.pkl'), 
             os.path.join(os.path.dirname(__file__), 'scaler.pkl')),  # Same directory as script
        ]
        
        model_loaded = False
        for mp, sp in possible_paths:
            try:
                # Load pre-trained model using pickle
                with open(mp, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load pre-trained scaler and column information
                with open(sp, 'rb') as f:
                    self.scaler = pickle.load(f)
                    self.numerical_cols = pickle.load(f)
                    self.categorical_cols = pickle.load(f)
                
                self.model_loaded = True
                model_loaded = True
                st.success("Model loaded successfully!")
                break
                
            except FileNotFoundError:
                continue
            except Exception as e:
                st.error(f"Error loading model: {e}")
                break
        
        if not model_loaded:
            st.error("Model files not found!")
            st.info("Please ensure model.pkl and scaler.pkl are in the project root directory or flask/ directory")
    
    def validate_input(self, feature_name: str, value: float) -> Tuple[bool, Optional[str]]:
        """
        Function: Validates user input against medical ranges.
        
        This function demonstrates:
        - Input validation logic
        - Dictionary data structure usage (VALIDATION_RANGES)
        - Tuple return type for multiple values
        
        Args:
            feature_name: Name of the feature to validate
            value: Input value to validate
            
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        # Dictionary lookup: Check if feature exists in validation ranges
        if feature_name not in VALIDATION_RANGES:
            return True, None  # Unknown feature, skip validation
        
        # Extract min and max values from dictionary
        min_val, max_val = VALIDATION_RANGES[feature_name]
        
        # Validation logic
        if value < min_val or value > max_val:
            error_msg = f"{feature_name} must be between {min_val} and {max_val}"
            return False, error_msg
        
        return True, None
    
    def create_categorical_features(self, bmi: float, insulin: float, glucose: float) -> np.ndarray:
        """
        Function: Creates categorical features from numerical inputs.
        
        This function demonstrates:
        - Feature engineering logic
        - NumPy array creation
        - List iteration for categorical feature encoding
        
        Args:
            bmi: Body Mass Index value
            insulin: Insulin level (μU/ml)
            glucose: Glucose level (mg/dL)
            
        Returns:
            NumPy array of one-hot encoded categorical features
        """
        # BMI category classification (based on WHO standards)
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi <= 24.9:
            bmi_cat = "Normal"
        elif bmi <= 29.9:
            bmi_cat = "Overweight"
        elif bmi <= 34.9:
            bmi_cat = "Obesity 1"
        elif bmi <= 39.9:
            bmi_cat = "Obesity 2"
        else:
            bmi_cat = "Obesity 3"
        
        # Insulin category (normal range: 16-166 μU/ml)
        insulin_cat = "Normal" if 16 <= insulin <= 166 else "Abnormal"
        
        # Glucose category (based on medical thresholds)
        if glucose <= 70:
            glucose_cat = "Low"
        elif glucose <= 99:
            glucose_cat = "Normal"
        elif glucose <= 126:
            glucose_cat = "Overweight"
        else:
            glucose_cat = "High"
        
        # NumPy array: Initialize categorical features array
        categorical_features = np.zeros(len(self.categorical_cols))
        
        # List iteration: Create one-hot encoded features
        for i, col in enumerate(self.categorical_cols):
            if 'NewBMI' in col:
                col_cat = col.replace('NewBMI_', '')
                if col_cat == bmi_cat:
                    categorical_features[i] = 1
            elif 'NewInsulinScore' in col:
                col_cat = col.replace('NewInsulinScore_', '')
                if col_cat == insulin_cat:
                    categorical_features[i] = 1
            elif 'NewGlucose' in col:
                col_cat = col.replace('NewGlucose_', '')
                if col_cat == glucose_cat:
                    categorical_features[i] = 1
        
        return categorical_features
    
    def prepare_features(self, user_inputs: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Function: Prepares features for model prediction.
        
        This function demonstrates:
        - Feature preprocessing pipeline
        - NumPy array operations
        - Dictionary data structure usage
        - Integration of numerical and categorical features
        
        Args:
            user_inputs: Dictionary containing user input values
            
        Returns:
            NumPy array of prepared features, or None if error occurs
        """
        try:
            # List comprehension: Extract numerical features in correct order
            numerical_features = np.array([[
                user_inputs['Pregnancies'],
                user_inputs['Glucose'],
                user_inputs['BloodPressure'],
                user_inputs['SkinThickness'],
                user_inputs['Insulin'],
                user_inputs['BMI'],
                user_inputs['DiabetesPedigreeFunction'],
                user_inputs['Age']
            ]])
            
            # NumPy array: Scale numerical features using pre-trained scaler
            numerical_scaled = self.scaler.transform(numerical_features)
            
            # Create categorical features
            categorical_features = self.create_categorical_features(
                user_inputs['BMI'],
                user_inputs['Insulin'],
                user_inputs['Glucose']
            )
            
            # NumPy array: Concatenate numerical and categorical features
            final_features = np.concatenate([numerical_scaled[0], categorical_features]).reshape(1, -1)
            
            return final_features
            
        except Exception as e:
            st.error(f"Error preparing features: {e}")
            return None
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, str]:
        """
        Function: Makes diabetes prediction using the trained model.
        
        This function demonstrates:
        - Model prediction using predict_proba
        - Probability calculation
        - Risk level assessment
        
        Args:
            features: NumPy array of prepared features
            
        Returns:
            Tuple of (prediction: int, probability: float, risk_level: str)
        """
        # Model prediction: Get class prediction (0 or 1)
        prediction = self.model.predict(features)[0]
        
        # Model prediction: Get probability scores using predict_proba
        prediction_proba = self.model.predict_proba(features)[0]
        
        # Probability calculation: Get diabetes probability
        diabetes_probability = prediction_proba[1] * 100
        
        # Risk level assessment based on probability
        if prediction == 1:
            if diabetes_probability > 70:
                risk_level = "High"
            else:
                risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return int(prediction), diabetes_probability, risk_level
    
    def run(self):
        """
        Method: Main application interface using Streamlit.
        
        This method demonstrates:
        - Streamlit UI components
        - User input collection
        - Integration of all functions
        - Error handling in UI context
        """
        # Streamlit: Page configuration with futuristic dark theme
        st.set_page_config(
            page_title="Metabolic Profile Analyzer - Neural Prediction Engine",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for futuristic dark theme with neon accents
        st.markdown("""
        <style>
        /* Dark theme base */
        .stApp {
            background: linear-gradient(180deg, #0a0a0f 0%, #000000 100%);
        }
        
        /* Main header - System title */
        .system-header {
            font-size: 3.5rem;
            font-weight: 900;
            color: #00ffff;
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 1rem;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            letter-spacing: -2px;
        }
        
        .system-subtitle {
            font-size: 1rem;
            color: #00d4ff;
            text-align: center;
            margin-bottom: 2rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            opacity: 0.8;
        }
        
        /* Data node styling for inputs */
        .data-node {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .data-node:hover {
            border-color: rgba(0, 255, 255, 0.6);
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        }
        
        .data-node-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }
        
        .data-node-value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            font-family: 'Courier New', monospace;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        }
        
        /* System alert for results */
        .system-alert {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
            border: 2px solid;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem 0;
        }
        
        .alert-high {
            border-color: #ff00ff;
            box-shadow: 0 0 30px rgba(255, 0, 255, 0.4);
        }
        
        .alert-moderate {
            border-color: #ffaa00;
            box-shadow: 0 0 30px rgba(255, 170, 0, 0.4);
        }
        
        .alert-low {
            border-color: #00ff41;
            box-shadow: 0 0 30px rgba(0, 255, 65, 0.4);
        }
        
        .risk-level-text {
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            margin: 1rem 0;
            text-transform: uppercase;
            letter-spacing: 4px;
        }
        
        .risk-high { color: #ff00ff; text-shadow: 0 0 20px rgba(255, 0, 255, 0.6); }
        .risk-moderate { color: #ffaa00; text-shadow: 0 0 20px rgba(255, 170, 0, 0.6); }
        .risk-low { color: #00ff41; text-shadow: 0 0 20px rgba(0, 255, 65, 0.6); }
        
        /* Probability display - large numbers */
        .probability-display {
            font-size: 4rem;
            font-weight: 900;
            color: #00ffff;
            text-align: center;
            font-family: 'Courier New', monospace;
            text-shadow: 0 0 25px rgba(0, 255, 255, 0.7);
            margin: 1rem 0;
        }
        
        /* Metric cards with neon glow */
        .metric-card-future {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #00ffff;
            font-family: 'Courier New', monospace;
            text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        }
        
        /* Signal strength bar visualization */
        .signal-bar {
            height: 40px;
            background: linear-gradient(90deg, #00ff41 0%, #ffaa00 50%, #ff00ff 100%);
            border-radius: 4px;
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .signal-fill {
            height: 100%;
            background: rgba(0, 255, 255, 0.8);
            transition: width 1s ease;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #0a0a0f 0%, #000000 100%);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
            border: 2px solid #00ffff;
            color: #00ffff;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.2) 0%, rgba(0, 212, 255, 0.2) 100%);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.5);
            transform: translateY(-2px);
        }
        
        /* Disclaimer styling */
        .disclaimer-minimal {
            font-size: 0.7rem;
            color: #6b7280;
            text-align: right;
            padding: 1rem;
            border-top: 1px solid rgba(107, 114, 128, 0.2);
            margin-top: 2rem;
        }
        
        /* Improved text readability */
        p, li, div {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        }
        
        /* Better contrast for readability */
        .readable-text {
            color: #e5e7eb;
            font-size: 1rem;
            line-height: 1.7;
        }
        
        /* Processing indicator */
        .processing-indicator {
            text-align: center;
            color: #00ffff;
            font-size: 1.2rem;
            margin: 2rem 0;
            text-transform: uppercase;
            letter-spacing: 3px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Futuristic System Header
        st.markdown('<div class="system-header">METABOLIC PROFILE ANALYZER</div>', unsafe_allow_html=True)
        st.markdown('<div class="system-subtitle">Neural Prediction Engine v2.0 | Real-time Risk Assessment</div>', unsafe_allow_html=True)
        
        # System Status Indicator
        if self.model_loaded:
            status_col1, status_col2, status_col3 = st.columns([1, 2, 1])
            with status_col2:
                st.markdown("""
                <div style='text-align: center; padding: 1rem; background: rgba(0, 255, 65, 0.1); 
                            border: 1px solid rgba(0, 255, 65, 0.3); border-radius: 8px; margin-bottom: 2rem;'>
                    <span style='color: #00ff41; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;'>
                        ⚡ System Status: READY | Engine: ACTIVE
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(255, 0, 0, 0.1); 
                        border: 1px solid rgba(255, 0, 0, 0.3); border-radius: 8px; margin-bottom: 2rem;'>
                <span style='color: #ff0000; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;'>
                    ⚠️ System Error: Model Not Loaded
                </span>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Main Layout: Split screen - Input Zone (Left) and Visualization Zone (Right)
        main_col1, main_col2 = st.columns([1, 1.5])
        
        with main_col1:
            st.markdown("""
            <div style='padding: 1rem; border-right: 1px solid rgba(0, 255, 255, 0.2); min-height: 600px;'>
                <h3 style='color: #00ffff; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px; 
                           margin-bottom: 1.5rem; border-bottom: 1px solid rgba(0, 255, 255, 0.3); padding-bottom: 0.5rem;'>
                    Input Parameters
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Dictionary: Store user inputs
        user_inputs = {}
        
        # Sidebar: Data Entry Zone with futuristic styling
        st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
                    border: 1px solid rgba(0, 255, 255, 0.3); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;'>
            <h3 style='color: #00ffff; margin: 0 0 0.5rem 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px;'>
                Scan Metabolic Profile
            </h3>
            <p style='color: #6b7280; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>
                Enter diagnostic measurements
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input fields styled as data nodes
        for feature_name in FEATURE_ORDER:
            min_val, max_val = VALIDATION_RANGES[feature_name]
            
            st.sidebar.markdown(f"""
            <div class="data-node">
                <div class="data-node-label">{feature_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            user_inputs[feature_name] = st.sidebar.number_input(
                label=feature_name,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.1 if feature_name in ['BMI', 'DiabetesPedigreeFunction'] else 1.0,
                help=f"Range: {min_val} - {max_val}",
                key=f"input_{feature_name}",
                label_visibility="collapsed"
            )
        
        # Primary action button - System language
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.sidebar.button(
            "⚡ RUN PREDICTION ENGINE", 
            type="primary", 
            use_container_width=True,
            help="Execute neural prediction analysis"
        )
        
        # Sidebar footer
        st.sidebar.markdown("""
        <div class="disclaimer-minimal">
            <strong>Neural Engine v2.0</strong><br>
            XGBoost Classifier | 89.47% Accuracy
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization Zone (Right side) - Results appear here
        with main_col2:
            if predict_button:
                # Error handling: Validate all inputs
                validation_errors = []
                for feature_name, value in user_inputs.items():
                    is_valid, error_msg = self.validate_input(feature_name, value)
                    if not is_valid:
                        validation_errors.append(error_msg)
                
                # Display validation errors if any
                if validation_errors:
                    st.markdown("""
                    <div style='background: rgba(255, 0, 0, 0.1); border: 2px solid #ff0000; 
                                border-radius: 12px; padding: 1.5rem; margin: 2rem 0;'>
                        <h3 style='color: #ff0000; margin-top: 0;'>⚠️ INPUT VALIDATION ERROR</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    for error in validation_errors:
                        st.error(f"• {error}")
                else:
                    # Function call: Prepare features (ML logic unchanged)
                    with st.spinner(""):
                        st.markdown('<div class="processing-indicator">⚡ Processing Metabolic Data...</div>', unsafe_allow_html=True)
                        features = self.prepare_features(user_inputs)
                    
                    if features is not None:
                        # Function call: Make prediction (ML logic unchanged)
                        prediction, probability, risk_level = self.predict(features)
                        
                        # System Alert - Risk Assessment Display
                        alert_class = f"alert-{risk_level.lower()}"
                        risk_color_class = f"risk-{risk_level.lower()}"
                        
                        st.markdown(f"""
                        <div class="system-alert {alert_class}">
                            <div class="risk-level-text {risk_color_class}">{risk_level} RISK</div>
                            <div style='text-align: center; color: #b0b0b0; font-size: 0.9rem; text-transform: uppercase; 
                                       letter-spacing: 2px; margin-bottom: 2rem;'>
                                Risk Signal Detected
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Large Probability Display
                        st.markdown(f"""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <div class="metric-label">Diabetes Probability</div>
                            <div class="probability-display">{probability:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Signal Strength Visualization
                        st.markdown("""
                        <div style='margin: 2rem 0;'>
                            <div class="metric-label" style='text-align: center; margin-bottom: 0.5rem;'>Risk Signal Strength</div>
                            <div class="signal-bar">
                                <div class="signal-fill" style='width: {}%;'></div>
                            </div>
                        </div>
                        """.format(probability), unsafe_allow_html=True)
                        
                        # Metrics Grid
                        metric_col1, metric_col2 = st.columns(2)
                        
                        with metric_col1:
                            confidence = max(probability, 100-probability)
                            st.markdown(f"""
                            <div class="metric-card-future">
                                <div class="metric-label">Prediction Confidence</div>
                                <div class="metric-value">{confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col2:
                            no_diabetes_prob = 100 - probability
                            st.markdown(f"""
                            <div class="metric-card-future">
                                <div class="metric-label">No Diabetes Probability</div>
                                <div class="metric-value">{no_diabetes_prob:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # System Message - Readable Format
                        if prediction == 1:
                            st.markdown("""
                            <div style='background: rgba(255, 0, 255, 0.1); border: 1px solid rgba(255, 0, 255, 0.3);
                                       border-radius: 8px; padding: 1.5rem; margin: 2rem 0;'>
                                <p style='color: #ff00ff; margin: 0 0 0.5rem 0; text-align: center; font-size: 1.1rem; font-weight: 600;'>
                                    SYSTEM ALERT: Diabetes Risk Indicators Detected
                                </p>
                                <p style='color: #d1d5db; margin: 0; text-align: center; font-size: 0.95rem; line-height: 1.6;'>
                                    Please consult a healthcare professional for proper diagnosis and medical advice.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='background: rgba(0, 255, 65, 0.1); border: 1px solid rgba(0, 255, 65, 0.3);
                                       border-radius: 8px; padding: 1.5rem; margin: 2rem 0;'>
                                <p style='color: #00ff41; margin: 0 0 0.5rem 0; text-align: center; font-size: 1.1rem; font-weight: 600;'>
                                    SYSTEM STATUS: No Diabetes Risk Detected
                                </p>
                                <p style='color: #d1d5db; margin: 0; text-align: center; font-size: 0.95rem; line-height: 1.6;'>
                                    Continue maintaining healthy lifestyle practices and regular health checkups.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                # Initial state - Awaiting data (readable format)
                st.markdown("""
                <div style='text-align: center; padding: 4rem 2rem; color: #9ca3af;'>
                    <div style='font-size: 3rem; margin-bottom: 1.5rem; opacity: 0.4;'>⚡</div>
                    <div style='font-size: 1.3rem; text-transform: uppercase; letter-spacing: 2px; 
                               margin-bottom: 0.75rem; color: #d1d5db; font-weight: 600;'>
                        Awaiting Metabolic Data
                    </div>
                    <div style='font-size: 1rem; color: #9ca3af; line-height: 1.6; max-width: 400px; margin: 0 auto;'>
                        Enter diagnostic parameters in the sidebar and click "RUN PREDICTION ENGINE" to analyze risk.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Medical Disclaimer - Readable Format (using Streamlit markdown, not raw HTML)
        st.markdown("---")
        with st.expander("⚠️ Medical & Research Disclaimer", expanded=False):
            st.markdown("### Educational Purpose Statement")
            st.markdown("""
            This application is developed as an **academic research project** for educational and research 
            purposes only. It demonstrates the application of machine learning techniques in medical diagnosis 
            prediction and is **NOT intended for clinical use**.
            """)
            
            st.markdown("### Medical Disclaimer")
            st.markdown("""
            • This tool is **NOT a substitute** for professional medical advice, diagnosis, or treatment  
            • Always consult qualified healthcare professionals for medical decisions  
            • Do not rely solely on this prediction for health-related decisions  
            • The model accuracy is approximately 90%, not 100%  
            • Regular medical checkups are essential for proper health management
            """)
            
            st.markdown("### Research Limitations")
            st.markdown("""
            **Dataset Limitation:** Model trained on Pima Indian heritage females (age ≥ 21 years)  
            **Generalization:** Results may not generalize to all populations or demographics  
            **Sample Size:** Training dataset consists of 768 samples  
            **Feature Scope:** Limited to 8 diagnostic measurements
            """)
            
            st.markdown("### Ethical Considerations")
            st.markdown("""
            This research tool should be used responsibly and in conjunction with professional medical 
            judgment. The developers and researchers are not liable for any medical decisions made based 
            on this prediction system.
            """)
        
        # Minimal footer disclaimer (always visible, readable)
        st.markdown("""
        <div style='text-align: center; padding: 1.25rem; margin-top: 2rem; 
                    border-top: 1px solid rgba(107, 114, 128, 0.2); background: rgba(0, 0, 0, 0.3);'>
            <p style='color: #9ca3af; font-size: 0.8rem; margin: 0; line-height: 1.7; font-family: Arial, sans-serif;'>
                <strong style='color: #ffaa00; font-size: 0.85rem;'>⚠️ Medical Disclaimer:</strong> 
                This is an academic research tool for educational purposes only. 
                <strong style='color: #ffffff;'>NOT a substitute</strong> for professional medical advice. 
                Model accuracy: ~90%. Always consult qualified healthcare professionals for medical decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN PROGRAM EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main program entry point.
    
    This demonstrates:
    - Program execution flow
    - Object instantiation (OOP)
    - Method calling
    """
    # Object-Oriented Programming: Create instance of DiabetesPredictionApp class
    app = DiabetesPredictionApp()
    
    # Method call: Run the Streamlit application
    app.run()

