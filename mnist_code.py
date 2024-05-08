from fastapi import FastAPI, UploadFile, File, Request, HTTPException
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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def wrapped_receive():
            return await receive()

        async def wrapped_send(message):
            await send(message)

        start_time = time.time()
        response = await self.app(scope, wrapped_receive, wrapped_send)
        end_time = time.time()

        # Measure request size
        request_size = len(scope.get("body") or b"")

        # Measure response size if the response object exists
        response_size = len(response.get("body") or b"") if response else 0

        # Calculate transfer rate
        duration = end_time - start_time
        transfer_rate = (request_size + response_size) / duration

        # Log or send metrics to Prometheus here
        print(f"Request Size: {request_size} bytes, Response Size: {response_size} bytes, Transfer Rate: {transfer_rate} bytes/sec")

        return response

# Load the model function
def load_model_function(path: str) -> Sequential:
    model = load_model(path)
    return model

# Function to preprocess the image
def preprocess_image(contents: bytes) -> np.ndarray:
    try:
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
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")

# Predict digit function
def predict_digit(model: Sequential, image_array: np.ndarray) -> str:
    try:
        # Flatten the input array to match the expected shape of the model
        flattened_array = image_array.flatten()
        # Reshape the flattened array to match the expected input shape of the model (None, 784)
        reshaped_array = flattened_array.reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(reshaped_array)
        digit = np.argmax(prediction)
        return str(digit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting digit: {str(e)}")

# Create FastAPI app
app = FastAPI()
app.add_middleware(MetricsMiddleware)

# Load the model
if len(sys.argv) != 2:
    raise ValueError("Usage: python mnist_api.py <model_path>")
model_path = sys.argv[1]
model = load_model_function(model_path)

# API endpoint for digit prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...), request:  Request = Request):
    try:
        start_time = time.time()  # Record start time
        # Read the uploaded file and process input
        contents = await file.read()
        # Preprocess the image
        image_array = preprocess_image(contents)
        digit = predict_digit(model, image_array)
        end_time = time.time()  # Record end time
        api_time = end_time - start_time  # Calculate total API time
        # Set gauge values
        input_length_gauge.set(len(image_array))
        api_time_gauge.set(api_time)
        tl_time_gauge.set((api_time * 1e6) / len(image_array)) # Calculate T/L time (microseconds per character)
        # Increment API usage counter
        api_usage_counter.labels(request.client.host).inc()
        return {"digit": digit}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST Digit Prediction API 1"}

# Expose Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Run the FastAPI app with Uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
