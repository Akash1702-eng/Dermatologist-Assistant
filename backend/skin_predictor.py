import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json # New Import

class SkinPredictor:
    def __init__(self):
        self.model = None
        self.diseases = {} # Will be loaded dynamically
        self.load_model()
    
    def load_model(self):
        """Load trained model and dynamic class mapping."""
        try:
            self.model = tf.keras.models.load_model('models/skin_model.h5')
            
            # Load dynamic class mapping
            with open('models/class_indices.json', 'r') as f:
                str_diseases = json.load(f)
                self.diseases = {int(k): v for k, v in str_diseases.items()}
                
            print(f"✅ Model loaded successfully! Classes: {len(self.diseases)}")
            return True
        except Exception as e:
            print(f"❌ Model or class mapping loading failed: {e}")
            return False
    
    def preprocess_image(self, image_array):
        """Preprocess image for prediction"""
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
        img_resized = cv2.resize(image_array, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
    
    def predict_image(self, image_path):
        """Predict skin disease from image file"""
        if self.model is None:
            print("Model not loaded.")
            return None
            
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            processed_img = self.preprocess_image(img_array)
            processed_batch = np.expand_dims(processed_img, axis=0)
            
            prediction = self.model.predict(processed_batch, verbose=0)[0]
            class_idx = np.argmax(prediction)
            
            # Determine top predictions for the output
            sorted_indices = np.argsort(prediction)[::-1]
            top_predictions = []
            for i in sorted_indices:
                top_predictions.append({
                    'disease': self.diseases.get(i, f"Class {i}"), 
                    'confidence': float(prediction[i])
                })
                
            predicted_disease = self.diseases.get(class_idx, "Unknown")
            
            return {
                'disease': predicted_disease,
                'confidence': float(prediction[class_idx]),
                'severity': self.determine_severity(predicted_disease),
                'top_predictions': top_predictions,
                'recommendations': self.get_recommendations(predicted_disease)
            }
        except Exception as e:
            print(f"❌ Prediction error for {image_path}: {e}")
            return None

    def determine_severity(self, disease_class):
        """Determine severity level (Simplified for a standalone predictor)"""
        high = ["Melanoma", "Basal Cell Carcinoma"]
        medium = ["Actinic Keratosis"]
        if disease_class in high:
            return "High"
        elif disease_class in medium:
            return "Medium"
        else:
            return "Low"

    def get_recommendations(self, disease_class):
        """Get general recommendations (Simplified for a standalone predictor)"""
        recommendations = {
            "Melanoma": ["Seek urgent specialist consultation", "Biopsy recommended", "Full body checkup"],
            "Basal Cell Carcinoma": ["Schedule dermatology appointment", "Consider surgical removal", "Monitor for recurrence"],
            "Actinic Keratosis": ["Dermatologist monitoring", "Sun protection essential", "Topical treatment options"],
            "Benign Keratosis": ["Generally harmless", "Cosmetic removal if desired", "Monitor for changes"],
            "Dermatofibroma": ["Generally benign, monitoring sufficient", "Surgical removal if symptomatic"],
            "Melanocytic Nevi": ["Regular self-monitoring (ABCDE rule)", "Annual dermatologist checkup"],
            "Vascular Lesion": ["Generally harmless", "Laser treatment available", "Sun protection recommended"]
        }
        return recommendations.get(disease_class, ["Consult healthcare professional"])

def main():
    predictor = SkinPredictor()
    
    if not predictor.model:
        return
    
    os.makedirs('test_images', exist_ok=True)
    
    test_folder = 'test_images'
    if os.path.exists(test_folder):
        test_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if test_files:
            print(f"🔍 Found {len(test_files)} test images:")
            for img_file in test_files:
                img_path = os.path.join(test_folder, img_file)
                print(f"\n📊 Analyzing: {img_file}")
                
                result = predictor.predict_image(img_path)
                if result:
                    print(f"   Disease: {result['disease']}")
                    print(f"   Confidence: {result['confidence']:.3f}")
                    print(f"   Severity: {result['severity']}")
                    print(f"   Top Predictions: {[f'{p['disease']}: {p['confidence']:.2f}' for p in result['top_predictions'][:3]]}")
                    print(f"   Recommendations: {', '.join(result['recommendations'])}")
        else:
            print("📁 No images found in 'test_images' folder.")
    
if __name__ == "__main__":
    main()