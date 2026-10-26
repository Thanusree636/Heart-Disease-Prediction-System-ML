"""
=============================================================================
Heart Disease Prediction - Flask REST API Backend
=============================================================================
Provides endpoints for health checking and model inference.
Accepts 13 clinical biomarkers via JSON, applies the trained SimpleImputer,
StandardScaler, and RandomForestClassifier, and returns risk assessments.
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) so frontend pages (running locally
# or served on a different port/protocol) can communicate seamlessly with Flask
CORS(app)

# Path to the serialized model artifact
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")

# Global containers for loaded artifacts
model_bundle = None
ml_model = None
scaler = None
imputer = None
feature_names = []

def load_artifacts():
    """Load the trained model, scaler, imputer, and feature names from the .pkl file."""
    global model_bundle, ml_model, scaler, imputer, feature_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Artifact file '{MODEL_PATH}' not found! "
            "Please run 'python train_model.py' first to train and generate the model."
        )
    print(f"[*] Loading ML model bundle from: {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    ml_model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    imputer = model_bundle["imputer"]
    feature_names = model_bundle["feature_names"]
    model_name = model_bundle.get("model_name", "Support Vector Machine (SVM)")
    print(f"[OK] Artifacts loaded successfully. Algorithm: {model_name}. Required features: {feature_names}")

# Load model artifacts at application startup
load_artifacts()


@app.route("/", methods=["GET"])
def index():
    """
    Health check endpoint.
    Returns status of the API and model information.
    """
    model_name = model_bundle.get("model_name", "Support Vector Machine (SVM)")
    return jsonify({
        "status": "online",
        "service": "Heart Disease Prediction API",
        "model": model_name,
        "features_required": feature_names,
        "endpoints": {
            "GET /": "Health check",
            "POST /predict": "Submit 13 patient biomarkers for heart disease risk prediction"
        }
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.
    Expects a JSON payload containing the 13 clinical features.
    Performs data validation, median imputation, standard scaling, and Random Forest inference.
    """
    try:
        # 1. Receive JSON payload
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({
                "error": "Invalid request body. Expected a JSON object with patient metrics."
            }), 400

        # 2. Validate presence and types of all 13 required features
        missing_fields = [feat for feat in feature_names if feat not in data or data[feat] is None or data[feat] == ""]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Parse numerical values safely
        cleaned_features = {}
        for feat in feature_names:
            try:
                val = float(data[feat])
                cleaned_features[feat] = val
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid numerical value for field '{feat}': received '{data[feat]}'"
                }), 400

        # 3. Convert input dictionary into pandas DataFrame with exact column ordering
        input_df = pd.DataFrame([cleaned_features], columns=feature_names)

        # 4. Apply saved median imputer
        imputed_array = imputer.transform(input_df)
        imputed_df = pd.DataFrame(imputed_array, columns=feature_names)

        # 5. Apply saved standard scaler
        scaled_array = scaler.transform(imputed_df)

        # 6. Perform inference with trained ML model (SVM)
        # Binary prediction: 0 = No Heart Disease, 1 = Heart Disease
        prediction_class = int(ml_model.predict(scaled_array)[0])

        # Probability estimates: [P(class 0), P(class 1)]
        probabilities = ml_model.predict_proba(scaled_array)[0]
        no_disease_prob = round(float(probabilities[0]) * 100, 1)
        disease_prob = round(float(probabilities[1]) * 100, 1)

        # 7. Formulate readable risk and result strings
        if prediction_class == 0:
            result_str = "No Heart Disease"
            risk_str = "Lower Risk"
        else:
            result_str = "Heart Disease Detected"
            risk_str = "Higher Risk"

        response_payload = {
            "prediction": prediction_class,
            "result": result_str,
            "risk": risk_str,
            "no_disease_probability": no_disease_prob,
            "disease_probability": disease_prob
        }

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "error": "An internal server error occurred during prediction.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    # Run the Flask development server on 127.0.0.1:5000
    print("[*] Starting Heart Disease Prediction Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
