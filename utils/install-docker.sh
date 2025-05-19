#!/bin/bash

# Update package list
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Add the current user to the docker group
sudo usermod -aG docker $USER

# # Download and install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# # Apply executable permissions to the binary
sudo chmod +x /usr/local/bin/docker-compose

# Verify Docker Compose installation
docker-compose --version

# Install Docker with NVIDIA support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify Docker with NVIDIA support
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || echo "NVIDIA GPU test failed. Please check that your GPU drivers are properly installed."

echo "Docker and Docker Compose have been installed successfully!"