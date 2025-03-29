# Use an official Python base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip3 install --no-cache-dir poetry

# Copy pyproject.toml and poetry.lock (if it exists)
COPY . /app/

# Install dependencies without installing the current project
RUN poetry install

# Set the default command to use Poetry to run your application
CMD ["/bin/bash"]
