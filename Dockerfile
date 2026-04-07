FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables with defaults
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV OPENAI_API_KEY=""
ENV HF_TOKEN=""
ENV PORT="7860"

# Default command starts API service for Spaces ping/reset checks
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
