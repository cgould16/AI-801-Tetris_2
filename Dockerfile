# Use the official TensorFlow GPU image as a base
FROM tensorflow/tensorflow:2.14.0-gpu

# Install curl to download get-pip.py
RUN apt-get update && apt-get install -y curl

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py
# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables for CUDA and cuDNN (if necessary)
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV ENVIRONMENT=development

# Expose port 80 (or the port your application uses)
EXPOSE 80

# Command to run your application
CMD ["python", "game.py"]
