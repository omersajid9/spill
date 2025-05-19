FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy clipsai first
COPY clipsai ./clipsai

# Copy and modify requirements to use local clipsai
COPY clipping/requirements.txt .

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Command to run when container starts
CMD ["python3", "./clipping/main.py"]