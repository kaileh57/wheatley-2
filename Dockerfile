# Wheatley 2.0 - Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper model during build to avoid runtime download
RUN python -c "import whisper; whisper.load_model('small.en')"

# Download sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY config.yaml.example config.yaml

# Create necessary directories
RUN mkdir -p data logs

# Expose ports
EXPOSE 8000 3000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "-m", "backend.main"] 