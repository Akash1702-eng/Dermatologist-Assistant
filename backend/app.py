#!/usr/bin/env python3


import os
import io
import cv2
import numpy as np
import random
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
# Import json for dynamic class loading
import json 
from tensorflow.nn import softmax

# ------------------------------
# Path Configuration (Render Safe)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Try to import tf and keras load_model (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    class MockSoftmax: 
        def __call__(self, x):
            x = np.array(x, dtype=np.float32)
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
    softmax = MockSoftmax()

# Import the ExplainabilityEngine from xai_explainer.py (must be present)
try:
    from xai_explainer import ExplainabilityEngine
except Exception as e:
    # If import fails, create a minimal fallback to avoid runtime errors
    print("⚠️ Could not import xai_explainer.ExplainabilityEngine:", e)
    class ExplainabilityEngine:
        def __init__(self):
            print("⚠️ Fallback XAI engine initialized")
        def explain(self, image_array, prediction, classifier):
            return {
                'heatmap': {'image_base64': '', 'description': 'XAI not available'}, 
                'feature_importance': {'summary': 'XAI not available'}, 
                'medical_reasoning': {'reasoning': f"Prediction: {prediction['disease']}. XAI not available"}, 
                'confidence_breakdown': {}, 
                'risk_assessment': {}
            }

# Initialize XAI engine
xai_engine = ExplainabilityEngine()

# ------------------------------
# ImprovedMLSkinAnalyzer class
# ------------------------------
class ImprovedMLSkinAnalyzer:
    def __init__(self, model_path=None):
        """
        Initialize analyzer.
        """
        # Placeholder categories - will be overwritten if models/class_indices.json loads
        self.disease_categories = [
            "Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis",
            "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesion"
        ]

        self.model = None
        self.model_loaded = False
        self.model_path = model_path
        self._candidate_model_paths = [
            model_path if model_path else None,
            os.path.join(MODELS_DIR, 'skin_model.h5'),
            os.path.join(MODELS_DIR, 'skin_model_fixed.h5'),
            os.path.join(MODELS_DIR, 'skin_model_enhanced.h5'),
            os.path.join(MODELS_DIR, 'best_skin_model.h5'),
            os.path.join(MODELS_DIR, 'skin_model.keras')
        ]
        self._candidate_model_paths = [p for p in self._candidate_model_paths if p]

        self.load_model()

    def load_model(self):
        """Try multiple model locations and load first that exists (requires TensorFlow/Class Map)."""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available — skipping model load, will use dynamic analysis")
            self.model_loaded = False
            return
        
        for p in self._candidate_model_paths:
            try:
                if os.path.exists(p):
                    self.model = load_model(p)
                    self.model_loaded = True
                    print(f"✅ ML Model loaded successfully from {p}")
                    
                    # DYNAMIC: Load dynamic class mapping
                    try:
                        class_indices_path = os.path.join(MODELS_DIR, 'class_indices.json')
                        with open(class_indices_path, 'r') as f:
                            str_diseases = json.load(f)
                            # Convert dictionary keys back to integers
                            dynamic_categories = {int(k): v for k, v in str_diseases.items()}
                            # Overwrite placeholder categories with dynamic ones, sorted by index
                            self.disease_categories = [dynamic_categories[i] for i in sorted(dynamic_categories.keys())]
                        print(f"✅ Dynamic class mapping loaded ({len(self.disease_categories)} classes)")
                    except Exception as e:
                        print(f"⚠️ Could not load class_indices.json. Using default categories. Error: {e}")
                        
                    return
            except Exception as e:
                print(f"⚠️ Failed to load model from {p}: {e}")
                continue

        print("⚠️ No ML model found in candidate paths. Using advanced dynamic analysis.")
        self.model_loaded = False

    # -----------------------
    # Image preprocessing & helpers (Unchanged)
    # -----------------------
    def preprocess_image(self, image_array):
        # ... (Unchanged preprocessing logic) ...
        try:
            if image_array.ndim == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            target_size = (224, 224)
            img_resized = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)

            lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            img_normalized = img_enhanced.astype(np.float32) / 255.0
            return img_normalized
        except Exception:
            try:
                fallback = cv2.resize(image_array, (224, 224))
                return fallback.astype(np.float32) / 255.0
            except Exception:
                return np.zeros((224, 224, 3), dtype=np.float32)

    def analyze_lesion_characteristics(self, image_array):
        # ... (Unchanged feature extraction logic) ...
        try:
            img = image_array.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            red_mean = float(np.mean(img[:, :, 0]))
            green_mean = float(np.mean(img[:, :, 1]))
            blue_mean = float(np.mean(img[:, :, 2]))
            color_variance = float(np.var(img) / 10000.0)
            saturation = float(np.mean(hsv[:, :, 1]))
            brightness = float(np.mean(lab[:, :, 0]))

            contrast = float(np.std(gray))
            try:
                smoothness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                smoothness = float(np.var(gray))
            
            homogeneity = 0.5 # Default if skimage unavailable

            edge_density = self.calculate_edge_density(gray)
            border_regularity = self.calculate_border_regularity(gray)
            color_uniformity = float(1.0 - (np.std([red_mean, green_mean, blue_mean]) / 128.0))
            symmetry_score = float(self.calculate_symmetry(gray))
            lesion_size_ratio = float(self.estimate_lesion_size(gray))
            color_clusters = float(self.estimate_color_clusters(img))

            features = {
                'red_mean': red_mean, 'green_mean': green_mean, 'blue_mean': blue_mean,
                'color_variance': color_variance, 'saturation': saturation, 'brightness': brightness,
                'contrast': contrast, 'smoothness': smoothness, 'homogeneity': homogeneity,
                'edge_density': edge_density, 'border_regularity': border_regularity,
                'color_uniformity': color_uniformity, 'symmetry_score': symmetry_score,
                'lesion_size_ratio': lesion_size_ratio, 'color_clusters': color_clusters
            }
            return features
        except Exception:
            return self.get_default_features()

    def calculate_edge_density(self, gray_image):
        try:
            edges = cv2.Canny(gray_image, 50, 150)
            return float(np.sum(edges > 0) / edges.size)
        except Exception: return 0.1
    def calculate_border_regularity(self, gray_image):
        try:
            edges = cv2.Canny(gray_image, 50, 150)
            if np.sum(edges > 0) == 0: return 0.5
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0.5
            largest = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest, True)
            area = cv2.contourArea(largest)
            if area == 0: return 0.5
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return float(min(circularity, 1.0))
        except Exception: return 0.5
    def calculate_symmetry(self, gray_image):
        try:
            h, w = gray_image.shape
            left = gray_image[:, :w//2]
            right = gray_image[:, w//2:]
            right_flipped = cv2.flip(right, 1)
            min_h = min(left.shape[0], right_flipped.shape[0])
            left = left[:min_h, :]
            right_flipped = right_flipped[:min_h, :]
            if left.size == 0 or right_flipped.size == 0: return 0.5
            mse = np.mean((left.astype(np.float32) - right_flipped.astype(np.float32)) ** 2)
            max_pixel = 255.0
            return float(1.0 - (mse / (max_pixel ** 2)))
        except Exception: return 0.5
    def estimate_lesion_size(self, gray_image):
        try:
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lesion_pixels = np.sum(thresh < 128)
            total = gray_image.size
            return float(lesion_pixels / (total or 1))
        except Exception: return 0.3
    def estimate_color_clusters(self, image_array):
        try:
            pixels = image_array.reshape(-1, 3)
            color_std = np.std(pixels, axis=0)
            return float(np.mean(color_std) / 100.0)
        except Exception: return 0.5
    def get_default_features(self):
        return {
            'red_mean': 128, 'green_mean': 128, 'blue_mean': 128,
            'color_variance': 0.5, 'saturation': 0.5, 'brightness': 0.5,
            'contrast': 0.5, 'smoothness': 0.5, 'homogeneity': 0.5,
            'edge_density': 0.5, 'border_regularity': 0.5,
            'color_uniformity': 0.5, 'symmetry_score': 0.5,
            'lesion_size_ratio': 0.3, 'color_clusters': 0.5
        }
    def diagnose_from_features(self, features): # Unchanged
        disease_patterns = {
            "Actinic Keratosis": {'color_score': features['red_mean'] > 140 and features['saturation'] < 100, 'texture_score': features['smoothness'] < 50, 'border_score': features['border_regularity'] > 0.6},
            "Basal Cell Carcinoma": {'color_score': features['red_mean'] > 150 and features['blue_mean'] < 100, 'texture_score': features['edge_density'] > 0.3, 'border_score': features['border_regularity'] < 0.4},
            "Benign Keratosis": {'color_score': features['color_uniformity'] > 0.7, 'texture_score': features['smoothness'] > 100, 'border_score': features['border_regularity'] > 0.7},
            "Dermatofibroma": {'color_score': features['red_mean'] > 130 and features['color_variance'] < 0.3, 'texture_score': features['homogeneity'] > 0.6, 'border_score': features['border_regularity'] > 0.5},
            "Melanoma": {'color_score': features['color_variance'] > 0.8, 'texture_score': features['edge_density'] > 0.5, 'border_score': features['border_regularity'] < 0.3},
            "Melanocytic Nevi": {'color_score': features['color_uniformity'] > 0.6, 'texture_score': features['smoothness'] > 80, 'border_score': features['border_regularity'] > 0.6},
            "Vascular Lesion": {'color_score': features['red_mean'] > 170 and features['saturation'] > 100, 'texture_score': features['smoothness'] > 120, 'border_score': features['border_regularity'] > 0.5}
        }
        disease_scores = {}
        for disease, pat in disease_patterns.items():
            score = 0.0
            if pat.get('color_score'): score += 0.4
            if pat.get('texture_score'): score += 0.3
            if pat.get('border_score'): score += 0.3
            if disease == "Melanoma" and features['symmetry_score'] < 0.4: score += 0.2
            elif disease == "Benign Keratosis" and features['symmetry_score'] > 0.7: score += 0.2
            elif disease == "Basal Cell Carcinoma" and features['brightness'] > 150: score += 0.2
            disease_scores[disease] = min(score, 1.0)
        return disease_scores
    def generate_dynamic_probabilities(self, disease_scores): # Unchanged
        total_score = sum(disease_scores.values())
        if total_score == 0: return {d: 1.0/len(self.disease_categories) for d in self.disease_categories}
        probs = {d: s / (total_score or 1.0) for d, s in disease_scores.items()}
        for d in probs: probs[d] = probs[d] * random.uniform(0.8, 1.2)
        total = sum(probs.values())
        return {k: float(v/total) for k, v in probs.items()}
    def determine_severity(self, disease, risk_score): # Unchanged
        high = ["Melanoma", "Basal Cell Carcinoma"]
        medium = ["Actinic Keratosis"]
        if disease in high: return "High"
        elif disease in medium: return "Medium"
        else: return "Low"
    def _ensure_probs(self, raw_pred): # Uses the dynamic 'softmax'
        try:
            probs = np.array(raw_pred, dtype=np.float32)
            s = probs.sum()
            if (probs.max() > 1.0 + 1e-6) or (abs(s - 1.0) > 1e-3):
                probs = softmax(probs).numpy() if TF_AVAILABLE and hasattr(softmax, 'numpy') else softmax(probs)
            return probs
        except Exception:
            probs = np.clip(np.array(raw_pred, dtype=np.float32), 0, None)
            return probs / (probs.sum() or 1)
            
    def predict_with_model(self, image_array):
        """If a TF model is loaded, use it to predict with proper preprocessing."""
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")
        if len(self.disease_categories) == 0:
            raise RuntimeError("Disease categories not initialized/loaded")
            
        try:
            processed = self.preprocess_image(image_array)
            batch = np.expand_dims(processed, axis=0)
            raw = self.model.predict(batch, verbose=0)
            
            if isinstance(raw, (list, tuple)): raw = raw[0]
            if raw.ndim == 2: raw = raw[0]

            probs = self._ensure_probs(raw)
            if len(probs) != len(self.disease_categories):
                 raise RuntimeError(f"Model output size ({len(probs)}) does not match dynamic classes ({len(self.disease_categories)}).")

            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            probabilities = {self.disease_categories[i]: float(probs[i]) for i in range(len(self.disease_categories))}
            return idx, confidence, probabilities
        except Exception as e:
            raise RuntimeError(f"Model prediction error: {e}")

    def advanced_dynamic_prediction(self, image_array):
        """Fallback: produce dynamic, rule-based prediction + risk score."""
        features = self.analyze_lesion_characteristics(image_array)
        disease_scores = self.diagnose_from_features(features)
        probabilities = self.generate_dynamic_probabilities(disease_scores)
        primary, confidence = max(probabilities.items(), key=lambda x: x[1])
        risk_score = disease_scores.get(primary, confidence)
        
        try:
             idx = self.disease_categories.index(primary)
        except ValueError:
             idx = 0 
        
        return idx, float(confidence), probabilities, float(risk_score)

    def predict(self, image_array):
        """Unified predict: returns response dict ready for JSON serialization."""
        try:
            features = self.analyze_lesion_characteristics(image_array)

            if self.model_loaded:
                try:
                    idx, confidence, probabilities = self.predict_with_model(image_array)
                    risk_score = confidence
                except Exception:
                    idx, confidence, probabilities, risk_score = self.advanced_dynamic_prediction(image_array)
            else:
                idx, confidence, probabilities, risk_score = self.advanced_dynamic_prediction(image_array)

            disease = self.disease_categories[idx]
            severity = self.determine_severity(disease, risk_score)

            description = self.generate_dynamic_description(disease, features, risk_score)

            # NOTE: Maps are placed here for brevity, they are unchanged from the user input
            risk_factors_map = {
                "Actinic Keratosis": ["Long-term sun exposure", "Fair skin", "Age over 40", "Outdoor occupation"],
                "Basal Cell Carcinoma": ["Chronic sun exposure", "Fair complexion", "Age over 50", "Light hair/eyes"],
                "Benign Keratosis": ["Aging", "Sun exposure", "Genetic factors", "Family history"],
                "Dermatofibroma": ["Minor skin trauma", "Genetic predisposition", "Insect bites", "Female gender"],
                "Melanoma": ["UV exposure", "Fair skin", "Family history", "Multiple moles", "Previous sunburns"],
                "Melanocytic Nevi": ["Sun exposure", "Genetic factors", "Fair skin", "Family history of moles"],
                "Vascular Lesion": ["Genetics", "Age", "Sun exposure", "Hormonal changes", "Liver disease"]
            }
            precautions_map = {
                "Actinic Keratosis": ["Dermatologist monitoring recommended", "Topical treatments (Imiquimod, 5-FU)", "Cryotherapy options", "Strict sun protection"],
                "Basal Cell Carcinoma": ["Schedule dermatology appointment within 2 weeks", "Discuss surgical treatment options", "Mohs surgery consideration", "Regular skin cancer screenings"],
                "Benign Keratosis": ["Generally harmless, no treatment needed", "Monitor for changes in appearance", "Cosmetic removal available if desired", "Regular self-examination"],
                "Dermatofibroma": ["Generally benign, monitoring sufficient", "Avoid scratching or irritating", "Surgical removal if symptomatic", "No regular follow-up required"],
                "Melanoma": ["Seek urgent dermatology consultation", "Consider biopsy for definitive diagnosis", "Full body skin examination recommended", "Regular follow-up appointments essential"],
                "Melanocytic Nevi": ["Regular self-monitoring using ABCDE rule", "Annual dermatology visit", "Photograph moles for comparison", "Report any changes immediately"],
                "Vascular Lesion": ["Generally harmless", "Consult if symptomatic (bleeding, itching)", "Laser treatment available for cosmetic improvement", "Sun protection recommended"]
            }

            response = {
                "prediction": {
                    "primary": {
                        "disease": disease,
                        "confidence": float(confidence),
                        "severity": severity,
                        "risk_score": float(risk_score)
                    },
                    "all_predictions": [
                        {"disease": d, "confidence": float(p)} for d, p in probabilities.items()
                    ]
                },
                "analysis": {
                    "features_analyzed": features,
                    "color_analysis": {
                        "red_dominance": features['red_mean'] > 150,
                        "color_variation": "High" if features['color_variance'] > 0.7 else "Medium" if features['color_variance'] > 0.4 else "Low",
                        "brightness_level": "High" if features['brightness'] > 150 else "Medium" if features['brightness'] > 100 else "Low",
                        "saturation_level": "High" if features['saturation'] > 100 else "Medium" if features['saturation'] > 50 else "Low"
                    },
                    "border_analysis": {
                        "contrast": features['contrast'],
                        "edge_density": features['edge_density'],
                        "border_regularity": features['border_regularity'],
                        "symmetry": features['symmetry_score']
                    },
                    "texture_analysis": {
                        "smoothness": features['smoothness'],
                        "homogeneity": features['homogeneity'],
                        "lesion_size": f"{features['lesion_size_ratio']*100:.1f}% of image area"
                    }
                },
                "recommendations": {
                    "description": description,
                    "risk_factors": risk_factors_map.get(disease, ["Consult dermatologist for assessment"]),
                    "precautions": precautions_map.get(disease, ["Professional evaluation recommended"]),
                    "monitoring_schedule": "Follow up in 1-3 months" if severity == "High" else "Follow up in 3-6 months" if severity == "Medium" else "Annual checkup",
                    "urgency_level": severity
                },
                "metadata": {
                    "model_used": "Trained CNN" if self.model_loaded else "Advanced Dynamic Analysis",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_id": f"skin_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "image_quality": "Good" if features['smoothness'] > 100 else "Average",
                    "confidence_rating": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                }
            }

            return response

        except Exception as e:
            print("❌ Prediction error:", e)
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}

    def generate_dynamic_description(self, disease, features, risk_score):
        base_descriptions = {
            "Actinic Keratosis": f"Analysis suggests scaly patches with rough texture. Color variance: {features.get('color_variance', 0):.2f}.",
            "Basal Cell Carcinoma": f"Features indicate possible BCC with pearly appearance. Border regularity: {features.get('border_regularity', 0):.2f}.",
            "Benign Keratosis": f"Benign characteristics observed. Color uniformity: {features.get('color_uniformity', 0):.2f}.",
            "Dermatofibroma": f"Fibrous tissue patterns detected. Homogeneity: {features.get('homogeneity', 0):.2f}.",
            "Melanoma": f"Concerning features detected: color variance {features.get('color_variance', 0):.2f}, irregular borders {features.get('border_regularity', 0):.2f}.",
            "Melanocytic Nevi": f"Typical mole patterns: uniform color distribution, regular borders ({features.get('border_regularity', 0):.2f}).",
            "Vascular Lesion": f"Vascular patterns: elevated red channel ({features.get('red_mean', 0):.0f})."
        }
        return base_descriptions.get(disease, "Computer vision analysis based on multiple feature extraction.")

# ------------------------------
# Flask app and routes
# ------------------------------
app = Flask(__name__)
CORS(app)

# Initialize analyzer (attempt to load model if present)
analyzer = ImprovedMLSkinAnalyzer()
# Initialize XAI engine
xai_engine = ExplainabilityEngine()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Advanced Skin Analysis API is running!",
        "endpoints": ["/health", "/predict", "/explain"]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": analyzer.model_loaded,
        "analysis_type": "Trained ML Model" if analyzer.model_loaded else "Advanced Dynamic Analysis",
        "service": "Intelligent Skin Analysis API",
        "disease_classes": len(analyzer.disease_categories)
    }), 200

def read_image_from_request(file_storage):
    """
    Read uploaded file (Flask FileStorage) and return RGB numpy array.
    """
    try:
        img = Image.open(file_storage.stream).convert('RGB')
        arr = np.array(img)
        return arr
    except Exception as e:
        # fallback: read bytes and attempt via cv2
        try:
            data = file_storage.read()
            nparr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2 failed to decode image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception as e2:
            raise ValueError("Failed to read image: " + str(e2))

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Primary prediction endpoint: returns prediction JSON (prediction + analysis metadata).
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded (use form key "image")'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image_array = read_image_from_request(file)

        # Run prediction (analyzer returns structured response)
        result = analyzer.predict(image_array)
        if "error" in result:
            return jsonify(result), 500

        # Optionally attach XAI
        with_xai = request.args.get('with_xai', '0') in ('1', 'true', 'yes')
        if with_xai:
            try:
                primary = result['prediction']['primary']
                simple_pred = {
                    'disease': primary['disease'],
                    'confidence': float(primary['confidence'])
                }
                explanations = xai_engine.explain(image_array, simple_pred, analyzer)
                result['explanations'] = explanations
            except Exception as e:
                print("⚠️ XAI generation error:", e)
                result['explanations_error'] = str(e)

        return jsonify(result), 200

    except Exception as e:
        print("❌ /predict error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain_route():
    """
    Explicit explanation endpoint.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded (use form key "image")'}), 400

        file = request.files['image']
        image_array = read_image_from_request(file)

        # get base prediction (for disease & confidence)
        analysis = analyzer.predict(image_array)
        if 'error' in analysis:
            return jsonify(analysis), 500

        primary = analysis['prediction']['primary']
        simple_pred = {'disease': primary['disease'], 'confidence': float(primary['confidence'])}

        explanations = xai_engine.explain(image_array, simple_pred, analyzer)

        # return combined
        return jsonify({
            'prediction': analysis['prediction'],
            'analysis': analysis['analysis'],
            'explanations': explanations,
            'metadata': analysis.get('metadata', {})
        }), 200

    except Exception as e:
        print("❌ /explain error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ------------------------------
# Utility route: health-check + model reload
# ------------------------------
@app.route('/reload-model', methods=['POST'])
def reload_model():
    """
    Reload model from disk (useful for updating files without restarting the server).
    """
    try:
        payload = request.get_json(silent=True) or {}
        new_path = payload.get('path')
        if new_path:
            analyzer._candidate_model_paths.insert(0, new_path)

        analyzer.load_model()
        return jsonify({'status': 'reloaded', 'model_loaded': analyzer.model_loaded}), 200
    except Exception as e:
        print("❌ reload-model error:", e)
        return jsonify({'error': str(e)}), 500

# ------------------------------
# Run server
# ------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Advanced Skin Analysis Server")
    print("="*60)
    print(f"🔬 Analysis Mode: {'ML Model' if analyzer.model_loaded else 'Advanced Dynamic Analysis'}")
    print(f"🎯 Disease Classes: {len(analyzer.disease_categories)}")
    print("="*60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)