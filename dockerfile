# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to the outside world
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Default model path

# Run app.py when the container launches
CMD ["uvicorn", "api_Code:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
