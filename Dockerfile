FROM docker.io/python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health check
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY demo/ ./demo/
COPY app.py .
COPY startup.sh .
COPY web.config .

# Set environment variables
ENV FLASK_APP=app.py
ENV PORT=80

# Expose port
EXPOSE 80

# Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    # CMD curl -f http://localhost:80/health || exit 1

# Run the application
CMD ["python", "app.py"]
