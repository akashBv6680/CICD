# File: Dockerfile
# Use a slim Python base image for a small footprint
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (including train.py)
COPY . .

# Command to run when the container starts (our training script)
CMD ["python", "train.py"]
