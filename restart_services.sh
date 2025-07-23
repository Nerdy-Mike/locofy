#!/bin/bash

# UI Component Labeling System - Service Restart Script
# This script helps resolve import errors and cached container issues

echo "🔄 UI Component Labeling System - Service Restart"
echo "================================================="

# Stop existing containers
echo "1. Stopping existing containers..."
docker-compose down

# Remove any orphaned containers
echo "2. Cleaning up orphaned containers..."
docker-compose down --remove-orphans

# Rebuild containers without cache to ensure fresh environment
echo "3. Rebuilding containers (this may take a few minutes)..."
docker-compose build --no-cache

# Start services
echo "4. Starting services..."
docker-compose up -d

# Wait a moment for services to initialize
echo "5. Waiting for services to initialize..."
sleep 10

# Check service status
echo "6. Checking service status..."
docker-compose ps

echo ""
echo "✅ Service restart complete!"
echo ""
echo "🌐 Access points:"
echo "   - Frontend (Streamlit): http://localhost:8501"
echo "   - Backend (FastAPI):    http://localhost:8000"
echo "   - API Health Check:     http://localhost:8000/health"
echo ""
echo "📝 If you still see import errors:"
echo "   1. Check the logs: docker-compose logs ui-labeling-frontend"
echo "   2. Try rebuilding again: docker-compose build --no-cache"
echo "   3. Ensure all files are saved and synced"
echo ""
echo "🔧 Troubleshooting:"
echo "   - View frontend logs: docker-compose logs -f ui-labeling-frontend"
echo "   - View backend logs:  docker-compose logs -f ui-labeling-api"
echo "   - Interactive shell:  docker-compose exec ui-labeling-frontend bash" 