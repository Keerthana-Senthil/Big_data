from fastapi import FastAPI, UploadFile, File, Request
from keras.models import Sequential
from typing import List
import sys, io 
from PIL import Image
import numpy as np
import uvicorn
from keras.models import load_model
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
from PIL import ImageOps


# Load the model function
def load_model_function(path: str) -> Sequential:
    model = load_model(path)
    return model

# Function to preprocess the image
def preprocess_image(contents: bytes) -> np.ndarray:
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


# Predict digit function
def predict_digit(model: Sequential, image_array: np.ndarray) -> str:
    # Flatten the input array to match the expected shape of the model
    flattened_array = image_array.flatten()
    # Reshape the flattened array to match the expected input shape of the model (None, 784)
    reshaped_array = flattened_array.reshape(1, -1)
    
    # Perform prediction
    prediction = model.predict(reshaped_array)
    digit = np.argmax(prediction)
    return str(digit)
# Create FastAPI app
app = FastAPI()

# Load the model
if len(sys.argv) != 2:
    print("Usage: python mnist_api.py <model_path>")
    sys.exit(1)
model_path = sys.argv[1]
model = load_model_function(model_path)

# Create counter to track API usage from different client IP addresses
api_usage_counter = Counter('api_usage', 'API usage from different client IP addresses', ['client_ip'])

# Create gauges to monitor API metrics
input_length_gauge = Gauge('input_length', 'Length of input text')
api_time_gauge = Gauge('api_time', 'Total time taken by the API')
tl_time_gauge = Gauge('tl_time', 'Effective processing time (microseconds per character)')

# API endpoint for digit prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...), request:  Request = Request):
    start_time = time.time()  # Record start time

    # Read the uploaded file and process input
    contents = await file.read()
    
    # Preprocess the image

    image_array = preprocess_image(contents)
    digit = predict_digit(model, image_array)
    end_time = time.time()  # Record end time
    api_time = end_time - start_time  # Calculate total API time
    input_length = image_array.shape[0]  # Get length of input text

    # Calculate T/L time (microseconds per character)
    tl_time = (api_time * 1e6) / input_length

    # Set gauge values
    input_length_gauge.set(input_length)
    api_time_gauge.set(api_time)
    tl_time_gauge.set(tl_time)

    # Get client IP address
    client_ip = request.client.host
    # Increment API usage counter
    api_usage_counter.labels(client_ip).inc()

    # Make prediction using the loaded model
    
    return {"digit": digit}
@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST Digit Prediction API 1"}

# Expose Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Run the FastAPI app with Uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
