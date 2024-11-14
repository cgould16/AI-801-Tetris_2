# Use the official TensorFlow GPU image as a base with CUDA 11.8
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory to /app
WORKDIR /app

# Install required system packages and dependencies for pygame
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils\
    curl \
    build-essential \
    python3-dev \
    fontconfig \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-mixer-dev \
    libgl1-mesa-glx \
    libx11-6 \
    libasound2-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip (if not included in the base image)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
 \

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

RUN pip install tensorflow[and-cuda]
RUN pip install cython wheel
# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add ROMs and application files
COPY roms/ ./roms/
COPY . .
RUN mkdir -p /app/tetris_regular_cnn_v1_nov_13/whole_model/

# Set environment variables for CUDA and cuDNN
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV XDG_RUNTIME_DIR /tmp/runtime-$USER
ENV AUDIODEV /dev/snd


# Expose application port
EXPOSE 80
RUN python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Run the application
CMD ["python", "game.py"]
CMD ["python", "tetris_ai.py"]