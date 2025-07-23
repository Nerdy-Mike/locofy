# UI Component Labeling System

A comprehensive tool for labeling UI components in design screenshots and evaluating AI-powered component detection using OpenAI GPT-4V.

## ✨ Features

### Phase 1 Implementation

- **📁 Image Management**: Upload and manage UI screenshots and design images
- **🎨 Interactive Annotation Tool**: Draw bounding boxes directly on images with mouse interaction
- **📝 Manual Annotation Entry**: Precise coordinate-based annotation for detailed work
- **🤖 AI-Powered Detection**: Automatic UI component detection using OpenAI GPT-4V
- **📊 Evaluation & Analytics**: Compare AI predictions against ground truth with IoU-based metrics
- **💾 Data Export**: Export annotations in structured JSON format
- **🔧 System Monitoring**: Health checks and system status monitoring

### Supported UI Components

- **Buttons**: Interactive button elements
- **Inputs**: Text input fields and form controls
- **Radio Buttons**: Radio button selections
- **Dropdowns**: Dropdown menus and select elements

## 🏗️ Architecture

The system follows a microservices architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (Port 8501)               │
│  ├── Interactive Drawing Canvas (streamlit-drawable-canvas)     │
│  ├── Image Upload & Management                                  │
│  ├── Annotation Interface                                       │
│  ├── AI Predictions Viewer                                      │
│  └── Evaluation Dashboard                                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP REST API
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Port 8000)                │
│  ├── Image Upload & Validation                                 │
│  ├── Annotation CRUD Operations                                │
│  ├── File Storage Management                                   │
│  └── LLM Integration Service                                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ OpenAI API
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OpenAI GPT-4V API                          │
│  └── Vision-based UI Element Detection                         │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (for AI predictions)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ui-labeling-system
```

### 2. Quick Start (Recommended)

Use the provided startup script for automatic setup:

```bash
./start.sh
```

This script will:
- Create `.env` file if missing
- Check Docker status
- Create necessary directories
- Start all services with proper configuration

### 3. Manual Setup (Alternative)

If you prefer manual setup, create a `.env` file in the project root:

```bash
# Required for AI predictions
OPENAI_API_KEY=your_openai_api_key_here

# Optional configurations
ENVIRONMENT=development
UPLOAD_DIR=/app/data/uploads
ANNOTATIONS_DIR=/app/data/annotations
```

### 4. Start the System

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 5. Access the Application

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 Usage Guide

### 1. Image Management

1. Navigate to the "🏠 Image Management" page
2. Upload UI screenshots or design images (PNG, JPG, GIF, BMP)
3. View your image library with thumbnails and metadata
4. Click "Edit" to start annotating an image

### 2. Interactive Annotation

The annotation tool provides three methods:

#### 🎨 Interactive Drawing
- Select a component type (button, input, radio, dropdown)
- Draw rectangles directly on the image by clicking and dragging
- Add annotator name and save all annotations at once

#### 📝 Manual Entry
- Enter precise coordinates for bounding boxes
- Useful for fine-tuning or when drawing is difficult
- Includes validation for bounds checking

#### 📊 Overview
- View annotation statistics and summary
- Manage existing annotations with edit/delete options
- Export annotations to JSON format
- Bulk operations for managing multiple annotations

### 3. AI Predictions

1. Go to the "🤖 AI Predictions" page
2. Select an image from your library
3. Click "Generate AI Predictions" to run OpenAI GPT-4V analysis
4. View side-by-side comparison of ground truth vs AI predictions
5. Analyze confidence scores and prediction accuracy

### 4. Evaluation & Statistics

- System-wide statistics dashboard
- Per-image evaluation metrics (Precision, Recall, F1-Score)
- Component type analysis and distribution charts
- Performance tracking over time

### 5. System Status

- Monitor API health and availability
- Check LLM service status
- View storage statistics and system information
- Test endpoint connectivity

## 🏗️ Development

### Project Structure

```
locofy/
├── adapters/
│   └── fastapi_adapter.py          # FastAPI application and routes
├── architecture/                    # Architecture documentation
├── docker/
│   ├── app.Dockerfile              # Backend container configuration
│   └── frontend.Dockerfile         # Frontend container configuration
├── frontend/
│   ├── components/                 # Reusable UI components
│   │   ├── annotation_canvas.py    # Interactive drawing canvas
│   │   └── image_viewer.py         # Image display with overlays
│   ├── utils/
│   │   └── api_client.py           # API client for backend communication
│   └── streamlit_app.py            # Main Streamlit application
├── models/
│   └── annotation_models.py        # Pydantic data models
├── services/
│   └── llm_service.py              # OpenAI GPT-4V integration
├── utils/
│   └── file_storage.py             # File management and storage
├── docker-compose.yml              # Multi-service orchestration
└── requirements.txt                # Python dependencies
```

### Local Development

1. **Backend Development**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run FastAPI with hot reload
   uvicorn adapters.fastapi_adapter:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Frontend Development**:
   ```bash
   # Run Streamlit with auto-reload
   streamlit run frontend/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
   ```

### Testing

```bash
# Test API endpoints
curl http://localhost:8000/health

# Test image upload
curl -X POST "http://localhost:8000/images/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.png"
```

## 📊 Data Models

### Annotation Structure

```json
{
  "id": "uuid-string",
  "image_id": "uuid-string", 
  "bounding_box": {
    "x": 100,
    "y": 50,
    "width": 200,
    "height": 30
  },
  "tag": "button",
  "confidence": 0.95,
  "annotator": "user",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Image Metadata

```json
{
  "id": "uuid-string",
  "filename": "ui_screenshot.png",
  "file_path": "/app/data/images/uuid.png",
  "file_size": 245760,
  "width": 1920,
  "height": 1080,
  "format": "PNG",
  "upload_time": "2024-01-01T00:00:00Z",
  "annotation_count": 5,
  "has_ai_predictions": true
}
```

## 🎯 Evaluation Metrics

The system uses IoU (Intersection over Union) based evaluation:

- **Precision**: `matches / total_predictions`
- **Recall**: `matches / total_annotations`  
- **F1-Score**: `2 * (precision * recall) / (precision + recall)`
- **IoU Threshold**: 0.5 (configurable)

Matches are determined by:
1. Same component type (tag)
2. IoU > threshold between bounding boxes

## 🧪 Command-Line Evaluation Tool

The system now includes comprehensive CLI tools for batch evaluation as required by the homework:

### Generate Test Data
```bash
# Generate 100 test annotation pairs
python3 src/cli/generate_test_data.py --output-dir ./test_datasets --count 100
```

### Batch Evaluation
```bash
# Evaluate folders of annotation files
python3 src/cli/evaluate_annotations.py \
    --ground-truth ./test_datasets/ground_truth \
    --predictions ./test_datasets/predictions \
    --output-json ./results.json
```

**Output Example:**
```
📊 UI COMPONENT ANNOTATION EVALUATION REPORT
📁 Files processed: 100/100 | 🎯 IoU threshold: 0.5

📈 OVERALL METRICS:
   Total Ground Truth Boxes: 456
   Correctly Predicted Boxes: 367
   Overall Precision: 0.868 | Overall Recall: 0.805 | Overall F1-Score: 0.835

🏷️  METRICS BY TAG:
Tag          GT Boxes   Pred Boxes   Correct  Precision  Recall   F1-Score
button       124        118          102      0.864      0.823    0.843
input        89         87           76       0.874      0.854    0.864
radio        134        125          108      0.864      0.806    0.834
dropdown     109        93           81       0.871      0.743    0.802
```

See `src/cli/README.md` for detailed documentation.

## 🔧 Configuration

### Environment Variables

| Variable          | Description                    | Default                  |
| ----------------- | ------------------------------ | ------------------------ |
| `OPENAI_API_KEY`  | OpenAI API key for GPT-4V      | Required for AI features |
| `ENVIRONMENT`     | Environment setting            | `development`            |
| `UPLOAD_DIR`      | Directory for uploaded images  | `/app/data/uploads`      |
| `ANNOTATIONS_DIR` | Directory for annotation files | `/app/data/annotations`  |

### Docker Configuration

The system uses multi-stage Docker builds for optimization:
- **Backend**: Python FastAPI with UV package manager
- **Frontend**: Streamlit with custom components
- **Volumes**: Persistent data storage for images and annotations

## 📈 Performance Considerations

- **Image Processing**: Images are resized for canvas display while maintaining aspect ratio
- **File Storage**: Organized directory structure with UUID-based naming
- **API Caching**: Streamlit resource caching for API client connections
- **Error Handling**: Comprehensive error handling with user-friendly messages

## 🔒 Security Features

- **File Validation**: Type and size restrictions on uploads
- **Input Sanitization**: Pydantic models for request validation
- **CORS Configuration**: Controlled cross-origin resource sharing
- **Path Security**: Secure file path handling to prevent directory traversal

## 🚧 Future Enhancements (Phase 2)

- **Multi-Annotator Support**: Conflict detection and resolution
- **Advanced Quality Metrics**: Inter-annotator agreement calculation
- **Batch Processing**: Handle multiple images simultaneously
- **Export Formats**: Support for COCO, YOLO, and other annotation formats
- **Advanced UI Components**: Support for more complex UI elements
- **Model Fine-tuning**: Custom model training on annotated data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

1. **Troubleshooting**: Check `TROUBLESHOOTING.md` for common issues and solutions
2. **Documentation**: Check the `/architecture` folder for detailed technical docs
3. **Issues**: Open an issue on the repository
4. **API Reference**: Visit http://localhost:8000/docs when running locally

---

**Built with ❤️ using FastAPI, Streamlit, and OpenAI GPT-4V** 