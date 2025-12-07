FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ingest.py .
COPY tools.py .
COPY agent.py .
COPY knowledge_base/ ./knowledge_base/

# Create directories for runtime data
RUN mkdir -p chroma_db

# Run ingestion to build vector database
RUN python ingest.py

# Set environment variable for Python buffering
ENV PYTHONUNBUFFERED=1

# Run the agent
CMD ["python", "agent.py"]
