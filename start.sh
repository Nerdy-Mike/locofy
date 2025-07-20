#!/bin/bash

# UI Component Labeling System - Startup Script
echo "🎨 Starting UI Component Labeling System..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating .env file from template..."
    cat > .env << EOF
# UI Component Labeling System - Environment Configuration

# OpenAI API Configuration (Required for AI predictions)
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
ENVIRONMENT=development

# File Storage Paths (Optional - defaults will be used if not specified)
UPLOAD_DIR=/app/data/uploads
ANNOTATIONS_DIR=/app/data/annotations

# API Configuration (Optional - used by frontend to connect to backend)
API_BASE_URL=http://ui-labeling-api:8000
EOF
    echo "✅ Created .env file. Please edit it with your OpenAI API key."
    echo "   Then run: docker-compose up --build"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if OpenAI API key is set
if grep -q "your_openai_api_key_here" .env; then
    echo "⚠️  Warning: Please set your OPENAI_API_KEY in .env file for AI predictions to work."
fi

# Create data directories if they don't exist
mkdir -p data/{images,annotations,metadata,predictions}

echo "🚀 Starting services with Docker Compose..."
docker-compose up --build

echo "🎉 System should be running at:"
echo "   Frontend: http://localhost:8501"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs" 