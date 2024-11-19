FROM tensorflow/tensorflow:2.10.0-gpu
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory to /app
WORKDIR /app

# Install required system packages and dependencies for pygame
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    curl \
    build-essential \
    python3.9 \
    python3-venv \
    python3-pip \
    python3-setuptools \
    nvidia-container-toolkit\
    libyaml-dev \
    python3-wheel \
    fontconfig \
    htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install cython tensorflow[and-cuda]==2.10

# Install additional Python dependencies
RUN pip install wheel
# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add ROMs and application files
COPY roms/ ./roms/
COPY . .
RUN mkdir -p /app/tetris_regular_cnn_v1_nov_13/whole_model/

# Set environment variables for CUDA and cuDNN
ENV XDG_RUNTIME_DIR=/tmp/runtime-$USER
ENV AUDIODEV=/dev/snd
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_VISIBLE_DEVICES=all
# Expose application port
EXPOSE 80
# Run the application
#CMD ["python3", "tetris_ai.py"]