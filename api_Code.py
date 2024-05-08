
from fastapi import FastAPI, UploadFile, File
from keras.models import Sequential
from typing import List
import sys, io 
from PIL import Image
import numpy as np
import uvicorn
from keras.models import load_model

# Your model loading and prediction functions will go here

def load_model_function(path: str) -> Sequential:  #Function to load the saved model
    model = load_model(path)
    return model

def predict_digit(model: Sequential, data_point: List[float]) -> str:
    #data_point serialised array of 784 elements representing the image is reshaped to predict the digit
    data_point = np.array(data_point).reshape(1, -1)
    prediction = model.predict(data_point) #Using the model to predict the result
    digit = np.argmax(prediction) #Selecting the class with the maximum probability
    return str(digit)#Returning the digit prediction

def format_image(contents: bytes) -> np.ndarray:
    """
    Resize the uploaded image to 28x28 grayscale and create a serialized array of 784 elements.

    Returns:
        np.ndarray: The serialized array of 784 elements representing the resized grayscale image.
    """
    # Convert bytes to Image object
    image = Image.open(io.BytesIO(contents))
    # Resize image to 28x28
    image = image.resize((28, 28))
    # Convert image to grayscale
    image = image.convert('L')
    # Convert image to numpy array
    image_array = np.array(image)
    # Flatten the image array to a 1D array
    serialized_array = image_array.flatten()
    # Normalize pixel values to range between 0 and 1
    serialized_array = serialized_array.astype('float32') / 255.0
    return serialized_array

print("Start")
    # Run the FastAPI app with Uvicorn
app = FastAPI()
    # Check if the user has provided the path to the model as a command line argument

    # Load the model
model_path = "best_mnist.h5"
model = load_model_function(model_path)
print("Model loaded")
    # API endpoint for digit prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
        # Convert image to 1D array of 784 elements
    serialized_array = format_image(contents)
        # Make prediction using the loaded model
    digit = predict_digit(model, serialized_array)
    return {"digit": digit} # Returning the prediction


    # Handler for root path
@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST Digit Prediction API 2"}#Printing message in root server to reassure that program is able to call fastapi


