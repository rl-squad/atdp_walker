# Use an official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy pyproject.toml and poetry.lock (if it exists)
COPY . /app/

# Install dependencies without installing the current project
RUN poetry install

# Set the default command to use Poetry to run your application
CMD ["bin/bash"]
