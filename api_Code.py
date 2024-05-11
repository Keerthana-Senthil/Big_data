import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from keras.models import Sequential
from typing import List
import sys, io 
from PIL import Image
import numpy as np
from keras.models import load_model
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter
from starlette.responses import Response

# Model loading and prediction functions 

def load_model_function(path: str) -> Sequential:
    """Function to load the saved model."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict_digit(model: Sequential, data_point: List[float]) -> str:
    """Predict the digit using the loaded model."""
    try:
        data_point = np.array(data_point).reshape(1, -1)
        prediction = model.predict(data_point)
        digit = np.argmax(prediction)
        return str(digit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting digit: {str(e)}")

def format_image(contents: bytes) -> np.ndarray:
    """Resize the uploaded image to 28x28 grayscale and create a serialized array of 784 elements."""
    try:
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
        # Validate the size of the serialized array
        if len(serialized_array) != 784:
            raise HTTPException(status_code=400, detail="Invalid image dimensions. Image must be 28x28 pixels.")
        return serialized_array
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error formatting image: {str(e)}")

print("Start")
# Run the FastAPI app with Uvicorn
app = FastAPI()

# Load the model
model_path = "best_mnist.h5"
model = load_model_function(model_path)
print("Model loaded")

api_usage_counter = Counter('api_usage', 'API usage from different client IP addresses', ['client_ip'])

# Create gauges to monitor API metrics
input_length_gauge = Gauge('input_length', 'Length of input text')
api_time_gauge = Gauge('api_time', 'Total time taken by the API')
tl_time_gauge = Gauge('tl_time', 'Effective processing time (microseconds per character)')

# API endpoint for digit prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = Request):
    """Endpoint to predict the digit from an uploaded image."""
    try:
        start_time = time.time()  # Start time

        # Get client IP address
        client_ip = request.client.host
        # Increment API usage counter
        api_usage_counter.labels(client_ip).inc()

        # Read the uploaded file
        contents = await file.read()
        
        # Convert image to 1D array of 784 elements
        serialized_array = format_image(contents)

        # Make prediction using the loaded model
        digit = predict_digit(model, serialized_array)

        end_time = time.time()  # Record end time
        api_time = end_time - start_time  # Calculate total API time
        input_length = serialized_array.shape[0]  # Get length of input text

        # Calculate T/L time (microseconds per character)
        tl_time = (api_time * 1e6) / input_length

        # Set gauge values
        input_length_gauge.set(input_length)
        api_time_gauge.set(api_time)
        tl_time_gauge.set(tl_time)

        # Logging client IP details
        print(f"Client IP: {client_ip}")

        return {"digit": digit}  # Returning the prediction
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Handler for root path
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the MNIST Digit Prediction API 2"}  # Printing message in root server to reassure that program is able to call fastapi

# Endpoint for serving Prometheus metrics
@app.get("/metrics")
async def metrics():
    """Endpoint to expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
