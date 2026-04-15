import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def predict_xray(img_path, model_path='saved_models/pneumonia_model.h5'):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalize
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    # Logic: Closer to 0 = Normal, Closer to 1 = Pneumonia
    label = "Pneumonia Detected" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Visualize
    plt.imshow(img)
    plt.title(f"Diagnosis: {label} (Confidence: {confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Run it manually on a test image
# Replace 'path_to_test_image.jpeg' with a real image path from your dataset
if __name__ == "__main__":
    test_image_path = "dataset/test/PNEUMONIA/person100_bacteria_475.jpeg" 
    predict_xray(test_image_path)