FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  stockfish \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy .env file
COPY .env .env

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "uvicorn", "adapters.fastapi_adapter:app", "--host", "0.0.0.0", "--port", "8080"] 