# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1-dev \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# 更新 pip
RUN pip install --no-cache-dir --upgrade pip


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "./AniTalker/code/webgui_copy2.py"]
