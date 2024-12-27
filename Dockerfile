# Use the official Python 3.12 slim image from Docker Hub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only necessary files into the container (excluding unnecessary files defined in .dockerignore)
COPY . /app

# Install system dependencies required for certain Python packages (e.g., AWS CLI, and others)
RUN apt-get update -y && apt-get install -y \
    awscli \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the environment variable to avoid buffering of Python output
ENV PYTHONUNBUFFERED=1

# Install Python dependencies from the requirements.txt file
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Streamlit app will use (default 8501)
EXPOSE 8501

# Set the command to run the Streamlit app (assuming your main app is app.py)
CMD ["streamlit", "run", "app.py"]