# Troubleshooting Guide

This guide helps resolve common issues when starting the UI Component Labeling System.

## 🔧 Backend Startup Issues

### Issue: Pydantic Warning about "model_name" field
```
Field "model_name" has conflict with protected namespace "model_".
```

**Solution**: This has been fixed by renaming `model_name` to `llm_model` and adding `protected_namespaces = ()` to the Pydantic config.

### Issue: Multiprocessing Errors
```
Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/local/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
```

**Solution**: This has been fixed by removing the `uvicorn.run()` call from the FastAPI adapter. Docker handles the uvicorn startup directly.

## 🚀 Quick Setup

### 1. Use the Startup Script (Recommended)
```bash
./start.sh
```

This script will:
- Check if Docker is running
- Create `.env` file if missing
- Create necessary data directories
- Start the services with proper configuration

### 2. Manual Setup
```bash
# 1. Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Create data directories
mkdir -p data/{images,annotations,metadata,predictions}

# 3. Start services
docker-compose up --build
```

## 🔍 Common Issues

### 1. Backend Won't Start
**Symptoms:**
- Error messages about Pydantic models
- Multiprocessing errors
- Import errors

**Solutions:**
- Ensure Docker is running: `docker info`
- Rebuild containers: `docker-compose down && docker-compose up --build`
- Check logs: `docker-compose logs ui-labeling-api`

### 2. Frontend Can't Connect to Backend
**Symptoms:**
- "Cannot connect to the API" message in Streamlit
- Connection refused errors

**Solutions:**
- Ensure backend is running: `curl http://localhost:8000/health`
- Check if ports are available: `netstat -an | grep 8000`
- Verify Docker network: `docker-compose ps`

### 3. AI Predictions Not Working
**Symptoms:**
- "LLM service not available" errors
- 503 Service Unavailable when generating predictions

**Solutions:**
- Set OPENAI_API_KEY in `.env` file
- Verify API key is valid: `curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models`
- Check backend logs for OpenAI API errors

### 4. File Upload Issues
**Symptoms:**
- Upload fails with 413 or 400 errors
- "Unsupported file type" messages

**Solutions:**
- Check file size (max 10MB)
- Ensure file type is supported (PNG, JPG, GIF, BMP)
- Verify data directories exist and are writable

## 📝 Environment Configuration

### Required Variables
```bash
OPENAI_API_KEY=your_api_key_here  # Required for AI predictions
```

### Optional Variables
```bash
ENVIRONMENT=development
UPLOAD_DIR=/app/data/uploads
ANNOTATIONS_DIR=/app/data/annotations
API_BASE_URL=http://ui-labeling-api:8000
```

## 🐛 Debug Mode

### Enable Verbose Logging
Add to your `.env` file:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Check Service Health
```bash
# Backend health
curl http://localhost:8000/health

# View backend logs
docker-compose logs -f ui-labeling-api

# View frontend logs  
docker-compose logs -f ui-labeling-frontend
```

### Inspect Running Containers
```bash
# List containers
docker-compose ps

# Enter backend container
docker-compose exec ui-labeling-api bash

# Enter frontend container
docker-compose exec ui-labeling-frontend bash
```

## 🔄 Reset and Clean Start

### Complete Reset
```bash
# Stop and remove containers
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Clean build cache
docker system prune -f

# Start fresh
docker-compose up --build
```

### Data Reset Only
```bash
# Remove only data (keeps containers)
rm -rf data/
mkdir -p data/{images,annotations,metadata,predictions}
```

## 📞 Getting Help

If issues persist:

1. **Check Logs**: Always check container logs first
   ```bash
   docker-compose logs ui-labeling-api
   docker-compose logs ui-labeling-frontend
   ```

2. **System Requirements**: 
   - Docker 20.0+
   - Docker Compose 1.27+
   - 4GB+ available RAM
   - Valid OpenAI API key

3. **Known Limitations**:
   - OpenAI API rate limits may affect AI predictions
   - Large images (>10MB) are not supported
   - Only 4 UI component types supported currently

4. **Report Issues**: Include logs and system information when reporting problems

## 🔧 Development Mode

### Running Without Docker
```bash
# Backend
pip install -r requirements.txt
uvicorn adapters.fastapi_adapter:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
streamlit run frontend/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

### Hot Reload
The Docker setup includes hot reload for development:
- Backend: Auto-reloads on Python file changes
- Frontend: Auto-reloads on Python file changes
- Volume mounts sync changes instantly 