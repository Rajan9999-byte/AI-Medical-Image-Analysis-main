import os
import matplotlib.pyplot as plt
from src.data_prep import get_data_generators
from src.model import build_medical_model

# 1. Setup paths
DATA_DIR = 'dataset/train' # Ensure you downloaded and unzipped the Kaggle data here
MODEL_SAVE_PATH = 'saved_models/pneumonia_model.h5'

# 2. Get Data
print("Loading data...")
train_gen, val_gen = get_data_generators(DATA_DIR)

# 3. Build Model
print("Building model...")
model = build_medical_model()

# 4. Train Model
print("Starting training...")
history = model.fit(
    train_gen,
    epochs=5, # Keep it low for initial testing
    validation_data=val_gen
)

# 5. Save Model
os.makedirs('saved_models', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# 6. Visualize Training
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/training_graph.png')
print("Training graph saved to outputs folder.")