# Use an official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy pyproject.toml and poetry.lock (if it exists)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies without installing the current project
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of your application code
COPY . /app

# Set the default command to use Poetry to run your application
CMD ["poetry", "run", "python", "test_walker.py"]
