import tensorflow as tf
import numpy as np
import cv2
import os
import json
# FIX APPLIED: Re-added the missing ImageDataGenerator import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2 
from tensorflow.keras.callbacks import EarlyStopping

class RealSkinAITrainer:
    def __init__(self):
        self.cnn_model = None
        self.diseases = {} 
        self.target_size = (224, 224)
        self.batch_size = 32
        
    def load_data_from_disk(self, data_dir='dataset'):
        """Loads data from the 'test' directory, splitting it into training and validation sets."""
        
        TRAIN_SOURCE_DIR = os.path.join(data_dir, 'test')
        
        if not os.path.exists(TRAIN_SOURCE_DIR) or not os.listdir(TRAIN_SOURCE_DIR):
            raise FileNotFoundError(
                f"🛑 Required REAL training data not found. Please ensure the directory '{TRAIN_SOURCE_DIR}' exists and contains class folders with images."
            )

        # 1. Data Augmentation and Rescaling for Training, with 20% validation split
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2, # 20% of images used for validation
            rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # 2. Load Training Data (80% of images)
        print(f"🔄 Loading training data from {TRAIN_SOURCE_DIR} (using 80% split)...")
        train_generator = train_datagen.flow_from_directory(
            TRAIN_SOURCE_DIR,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training', # Use the training subset
            shuffle=True
        )
        
        # 3. Load Validation Data (20% of images)
        print(f"🔄 Loading validation data from {TRAIN_SOURCE_DIR} (using 20% split)...")
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_SOURCE_DIR,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation', # Use the validation subset
            shuffle=False
        )
        
        # 4. Save Class Mapping
        self.diseases = {v: k for k, v in train_generator.class_indices.items()}
        print(f"✅ Classes detected: {list(self.diseases.values())}")
        
        return train_generator, validation_generator

    def create_transfer_model(self, num_classes):
        """Creates a model using ResNet50V2 (Transfer Learning) for high accuracy."""
        
        # Load ResNet50V2 base model
        base_model = ResNet50V2(
            input_shape=(224, 224, 3), 
            include_top=False, 
            weights='imagenet' 
        )
        
        # Freeze the base model layers 
        base_model.trainable = False 
        
        # Build the new classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(), 
            Dense(256, activation='relu'), 
            Dropout(0.5),
            Dense(num_classes, activation='softmax') # Dynamic output layer
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train_model(self):
        """Train CNN model using loaded real data."""
        print("🚀 Starting model training on REAL DATA (ResNet50V2 Transfer Learning)...")
        
        try:
            train_generator, validation_generator = self.load_data_from_disk()
        except Exception as e:
            print(f"🛑 Training aborted: {e}")
            return

        num_classes = train_generator.num_classes
        
        train_steps = train_generator.samples // self.batch_size
        val_steps = validation_generator.samples // self.batch_size
        
        if train_steps == 0 or val_steps == 0:
             print("🛑 Training aborted: Not enough images for one full batch (32 images total). Please add more images to your class folders.")
             return

        print("🧠 Training Transfer Learning model...")
        self.cnn_model = self.create_transfer_model(num_classes)
        
        # Define Early Stopping Callback
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )

        self.cnn_model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=50, 
            validation_data=validation_generator,
            validation_steps=val_steps,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✅ Training Complete!")
        self.save_model()
    
    def save_model(self):
        """Save trained model and class mapping."""
        os.makedirs('models', exist_ok=True)
        self.cnn_model.save('models/skin_model.h5') 
        print("💾 Model saved as 'models/skin_model.h5'")
        
        with open('models/class_indices.json', 'w') as f:
            str_diseases = {str(k): v for k, v in self.diseases.items()}
            json.dump(str_diseases, f)
        print("💾 Class mapping saved as 'models/class_indices.json'")
    
    def load_model(self):
        try:
            from tensorflow.keras.models import load_model
            self.cnn_model = load_model('models/skin_model.h5')
            with open('models/class_indices.json', 'r') as f:
                str_diseases = json.load(f)
                self.diseases = {int(k): v for k, v in str_diseases.items()}
            return True
        except Exception:
            return False

    def get_fallback_prediction(self):
        return {'disease': 'Consult Dermatologist', 'confidence': 0.5, 'all_probabilities': {}}
    
    def predict(self, image_array):
        if self.cnn_model is None:
            if not self.load_model():
                return self.get_fallback_prediction()
        try:
            processed_img = self._preprocess_image(image_array)
            processed_batch = np.expand_dims(processed_img, axis=0)
            prediction = self.cnn_model.predict(processed_batch, verbose=0)[0]
            class_idx = np.argmax(prediction)
            
            probabilities = {}
            for i, prob in enumerate(prediction):
                disease_name = self.diseases.get(i, f"Class {i}")
                probabilities[disease_name] = float(prob)
                
            return {'disease': self.diseases.get(class_idx, "Unknown Disease"), 'confidence': float(prediction[class_idx]), 'all_probabilities': probabilities}
        except Exception:
            return self.get_fallback_prediction()

    def _preprocess_image(self, img_array):
        img_resized = cv2.resize(img_array, self.target_size)
        return img_resized.astype(np.float32) / 255.0

def main():
    trainer = RealSkinAITrainer()
    trainer.train_model()
    
if __name__ == "__main__":
    main()