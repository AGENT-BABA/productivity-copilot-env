# Use a standard python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Set up working directory
WORKDIR /app

# Copy requirements first
COPY pyproject.toml /app/
COPY uv.lock /app/

# Install python dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY . /app

# Set port for HuggingFace Spaces
ENV PORT=7860
EXPOSE 7860

# Run the OpenEnv server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
