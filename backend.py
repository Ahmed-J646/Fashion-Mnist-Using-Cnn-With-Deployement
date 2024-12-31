from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (make sure it's saved after training)
model = tf.keras.models.load_model('fashion_mnist_cnn_model.h5')

# Class names (as in Fashion MNIST dataset)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Helper function to preprocess the uploaded image
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 as expected by the model
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add the channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Create prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    
    # Preprocess the image
    img_array = preprocess_image(image_data)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Return the class label and prediction confidence
    return {"class": class_names[predicted_class[0]], "confidence": prediction[0][predicted_class[0]]}

