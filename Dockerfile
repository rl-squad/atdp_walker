# GPU supported image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Installs Python dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip3 install --no-cache-dir poetry

# Copy in all files from current directory
COPY . /app/

# Install dependencies without installing the current project
RUN poetry install

# Runs bash inside container
CMD ["/bin/bash"]
