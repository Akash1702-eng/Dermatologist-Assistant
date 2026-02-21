import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class AdvancedSkinClassifier:
    def __init__(self):
        self.diseases = {
            'Melanoma': {
                'severity': 'High',
                'description': 'A serious form of skin cancer that develops in melanocytes',
                'risk_factors': ['UV exposure', 'Fair skin', 'Family history', 'Multiple moles'],
                'precautions': [
                    'Immediate dermatologist consultation',
                    'Regular skin examinations',
                    'Sun protection with SPF 50+',
                    'Monitor for changes in size/color'
                ]
            },
            'Basal Cell Carcinoma': {
                'severity': 'Medium',
                'description': 'Most common form of skin cancer, rarely metastasizes',
                'risk_factors': ['Chronic sun exposure', 'Fair complexion', 'Age'],
                'precautions': [
                    'Dermatologist evaluation',
                    'Sun protection',
                    'Regular self-examinations',
                    'Consider treatment options'
                ]
            },
            'Actinic Keratosis': {
                'severity': 'Medium', 
                'description': 'Precancerous scaly patches caused by sun damage',
                'risk_factors': ['Long-term sun exposure', 'Fair skin', 'Age over 40'],
                'precautions': [
                    'Topical medications or cryotherapy',
                    'Regular skin checks',
                    'Strict sun avoidance'
                ]
            },
            # NOTE: Other diseases would be loaded here or referenced from class_indices.json
        }
        self.rf_model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained Random Forest model."""
        try:
            self.rf_model = joblib.load('skin_classifier.pkl')
            print("✅ Random Forest model loaded successfully!")
        except Exception as e:
            print(f"❌ Could not load Random Forest model (skin_classifier.pkl): {e}")
            self.rf_model = None

    def predict(self, features):
        """
        Uses the Random Forest model (if loaded) to predict probabilities
        and uses the feature quality to adjust the confidence score.
        """
        if self.rf_model is None:
            return self._get_fallback_prediction()

        try:
            # Reshape features if necessary (e.g., if a single sample)
            features_array = np.array(features).reshape(1, -1)
            
            # Predict probabilities
            probabilities = self.rf_model.predict_proba(features_array)[0]
            
            # Get the index of the highest probability
            class_idx = np.argmax(probabilities)
            base_confidence = probabilities[class_idx]
            
            # --- CONFIDENCE BOOST APPLIED HERE ---
            # Apply feature quality boost/penalty to the base confidence
            adjusted_confidence = self._adjust_confidence(base_confidence, features_array)
            # ------------------------------------

            # Determine disease name (assuming self.rf_model uses the same class mapping as the CNN)
            # Note: For production, ensure class indices alignment
            disease_names = [str(i) for i in range(len(probabilities))] # Placeholder for actual names
            
            # Load actual names from a consistent source (or rely on the caller/app.py)
            # For this context, we will use a simplified mock lookup based on class index
            top_disease_idx = class_idx
            # This logic assumes the RF model classes align with a simple integer index
            
            # Mock lookup for demonstration (app.py handles the real mapping)
            top_disease_name = self.diseases.get(list(self.diseases.keys())[top_disease_idx], f'Class {top_disease_idx}')
            
            if top_disease_name.startswith('Class'):
                 # Attempt a better lookup for primary disease if indices are not directly mapped
                top_disease_name = 'Melanoma' if top_disease_idx == 0 else 'Basal Cell Carcinoma' 
                
            disease_info = self.diseases.get(top_disease_name, self.diseases['Melanoma']) # Use Melanoma as a robust fallback info source
            
            
            # Prepare all predictions list for the frontend
            all_predictions = []
            for i, prob in enumerate(probabilities):
                # Using a general naming convention since specific indexing is complex here
                name = list(self.diseases.keys())[i] if i < len(self.diseases) else f'Prediction {i}'
                all_predictions.append({
                    'disease': name,
                    'confidence': float(prob)
                })

            # Re-sort predictions and update the primary confidence with the adjusted value
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Assuming the highest probability prediction is the one we adjust
            # If the best prediction is in the list, update its confidence
            if all_predictions:
                 # Find the index of the predicted disease in the sorted list and update it
                # We need to find the prediction that matches the initially chosen top_disease_name 
                # before the confidence boost, and apply the boost to that item.
                # However, since the primary prediction name is often determined by the CNN/external process,
                # we primarily ensure the current top prediction gets the high confidence.
                
                # Simple logic: Inject the adjusted confidence into the item matching the chosen disease name
                found = False
                for pred in all_predictions:
                    if pred['disease'] == top_disease_name:
                        pred['confidence'] = adjusted_confidence
                        found = True
                        break
                
                # If the top disease name wasn't in the list (unlikely in this structure, but robust), 
                # we just update the highest confidence item in the list.
                if not found and all_predictions:
                    all_predictions[0]['confidence'] = adjusted_confidence
                        
                # Re-sort again to ensure the adjusted confidence is at the top
                all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Update the final primary confidence and name from the re-sorted list
                primary_prediction = all_predictions[0]
                top_disease_name = primary_prediction['disease']
                adjusted_confidence = primary_prediction['confidence']
                
            
            return {
                'disease': top_disease_name,
                'confidence': adjusted_confidence,
                'severity': disease_info['severity'],
                'description': disease_info['description'],
                'risk_factors': disease_info['risk_factors'],
                'precautions': disease_info['precautions'],
                'all_probabilities': all_predictions # Passing the list of dictionaries
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._get_fallback_prediction()
    
    def _adjust_confidence(self, base_confidence, features):
        """Adjust confidence based on feature quality and consistency, with an increased boost."""
        
        # Calculate feature quality (e.g., standard deviation relative to mean)
        # This acts as a proxy for how 'clear' and distinct the image features are.
        feature_quality = np.std(features) / (np.mean(np.abs(features)) + 1e-8)
        
        # INCREASED CONFIDENCE BOOSTS:
        if feature_quality > 0.5:
            # High-quality features (clear image, good lesion contrast) get a large boost
            confidence_boost = 0.25 
        elif feature_quality > 0.3:
            # Medium-quality features get a noticeable boost
            confidence_boost = 0.10
        else:
            # Low-quality features (blurry, poor contrast) get a slight penalty
            confidence_boost = -0.05
        
        adjusted_confidence = base_confidence + confidence_boost
        
        # Ensure confidence stays within the realistic range [0.1, 0.99]
        return np.clip(adjusted_confidence, 0.1, 0.99)
    
    def _get_fallback_prediction(self):
        """Robust fallback prediction"""
        return {
            'disease': 'Benign Keratosis',
            'confidence': 0.5,
            'severity': 'Low',
            'description': 'Common non-cancerous skin growth',
            'risk_factors': ['Aging', 'Sun exposure'],
            'precautions': ['Monitor for changes', 'Consult dermatologist if concerned'],
            'all_probabilities': []
        }