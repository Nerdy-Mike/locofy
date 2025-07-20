import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.annotation_models import (
    Annotation,
    AnnotationCreate,
    AnnotationUpdate,
    ImageMetadata,
    LLMPrediction,
)
from services.llm_service import LLMUIDetectionService
from utils.file_storage import FileStorageManager

# Initialize FastAPI app
app = FastAPI(
    title="UI Component Labeling API",
    description="API for labeling UI components in design screenshots",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
storage_manager = FileStorageManager()

# Initialize LLM service if API key is available
llm_service = None
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    llm_service = LLMUIDetectionService(openai_api_key)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "UI Component Labeling API",
        "version": "1.0.0",
        "endpoints": {
            "images": "/images",
            "annotations": "/annotations",
            "predictions": "/predictions",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = storage_manager.get_storage_stats()
    return {
        "status": "healthy",
        "llm_service_available": llm_service is not None,
        "storage_stats": stats,
    }


# Image Management Endpoints


@app.post("/images/upload", response_model=ImageMetadata)
async def upload_image(file: UploadFile = File(...)):
    """Upload a new image for labeling"""

    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}",
        )

    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB",
        )

    # Reset file pointer
    await file.seek(0)

    try:
        metadata = storage_manager.save_image(file.file, file.filename)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")


@app.get("/images", response_model=List[ImageMetadata])
async def list_images():
    """List all uploaded images"""
    return storage_manager.list_images()


@app.get("/images/{image_id}", response_model=ImageMetadata)
async def get_image_metadata(image_id: str):
    """Get metadata for a specific image"""
    metadata = storage_manager.get_image_metadata(image_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Image not found")
    return metadata


@app.get("/images/{image_id}/file")
async def get_image_file(image_id: str):
    """Get the actual image file"""
    image_path = storage_manager.get_image_path(image_id)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        path=str(image_path), media_type="image/jpeg", filename=f"{image_id}.jpg"
    )


@app.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """Delete an image and all associated data"""
    success = storage_manager.delete_image(image_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Image not found or could not be deleted"
        )

    return {"message": "Image deleted successfully"}


# Annotation Management Endpoints


@app.get("/images/{image_id}/annotations", response_model=List[Annotation])
async def get_annotations(image_id: str):
    """Get all annotations for an image"""
    # Verify image exists
    if not storage_manager.get_image_metadata(image_id):
        raise HTTPException(status_code=404, detail="Image not found")

    return storage_manager.get_annotations(image_id)


@app.post("/images/{image_id}/annotations", response_model=Annotation)
async def create_annotation(image_id: str, annotation: AnnotationCreate):
    """Create a new annotation for an image"""
    # Verify image exists
    if not storage_manager.get_image_metadata(image_id):
        raise HTTPException(status_code=404, detail="Image not found")

    # Get existing annotations
    existing_annotations = storage_manager.get_annotations(image_id)

    # Create new annotation
    new_annotation = Annotation(
        image_id=image_id,
        bounding_box=annotation.bounding_box,
        tag=annotation.tag,
        annotator=annotation.annotator,
    )

    # Add to existing annotations
    existing_annotations.append(new_annotation)

    # Save all annotations
    storage_manager.save_annotations(image_id, existing_annotations)

    return new_annotation


@app.put("/annotations/{annotation_id}", response_model=Annotation)
async def update_annotation(annotation_id: str, update: AnnotationUpdate):
    """Update an existing annotation"""
    # Find the annotation across all images (this is a simplified approach)
    # In a production system, you might want to index annotations differently

    all_images = storage_manager.list_images()
    for image_metadata in all_images:
        annotations = storage_manager.get_annotations(image_metadata.id)
        for i, annotation in enumerate(annotations):
            if annotation.id == annotation_id:
                # Update the annotation
                if update.bounding_box:
                    annotation.bounding_box = update.bounding_box
                if update.tag:
                    annotation.tag = update.tag
                if update.annotator:
                    annotation.annotator = update.annotator

                # Save updated annotations
                storage_manager.save_annotations(image_metadata.id, annotations)
                return annotation

    raise HTTPException(status_code=404, detail="Annotation not found")


@app.delete("/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation"""
    # Find and remove the annotation
    all_images = storage_manager.list_images()
    for image_metadata in all_images:
        annotations = storage_manager.get_annotations(image_metadata.id)
        for i, annotation in enumerate(annotations):
            if annotation.id == annotation_id:
                # Remove the annotation
                annotations.pop(i)
                storage_manager.save_annotations(image_metadata.id, annotations)
                return {"message": "Annotation deleted successfully"}

    raise HTTPException(status_code=404, detail="Annotation not found")


# LLM Prediction Endpoints


@app.post("/images/{image_id}/predict", response_model=LLMPrediction)
async def generate_predictions(image_id: str):
    """Generate LLM predictions for UI components in an image"""
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service not available. Check OPENAI_API_KEY configuration.",
        )

    # Verify image exists
    image_path = storage_manager.get_image_path(image_id)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        # Generate predictions
        predictions = llm_service.detect_ui_components(image_id, str(image_path))

        # Save predictions
        storage_manager.save_llm_predictions(predictions)

        return predictions
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating predictions: {str(e)}"
        )


@app.get("/images/{image_id}/predictions", response_model=Optional[LLMPrediction])
async def get_predictions(image_id: str):
    """Get existing LLM predictions for an image"""
    # Verify image exists
    if not storage_manager.get_image_metadata(image_id):
        raise HTTPException(status_code=404, detail="Image not found")

    predictions = storage_manager.get_llm_predictions(image_id)
    return predictions


# Evaluation and Statistics Endpoints


@app.get("/images/{image_id}/evaluation")
async def evaluate_predictions(image_id: str):
    """Evaluate LLM predictions against ground truth annotations"""
    # Get both annotations and predictions
    annotations = storage_manager.get_annotations(image_id)
    predictions_data = storage_manager.get_llm_predictions(image_id)

    if not predictions_data:
        raise HTTPException(
            status_code=404, detail="No predictions found for this image"
        )

    predictions = predictions_data.predictions

    # Simple evaluation metrics
    total_annotations = len(annotations)
    total_predictions = len(predictions)

    # Calculate IoU-based matches (simplified)
    matches = 0
    iou_threshold = 0.5

    for annotation in annotations:
        for prediction in predictions:
            if (
                annotation.tag == prediction.tag
                and annotation.bounding_box.calculate_iou(prediction.bounding_box)
                > iou_threshold
            ):
                matches += 1
                break

    precision = matches / total_predictions if total_predictions > 0 else 0
    recall = matches / total_annotations if total_annotations > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "image_id": image_id,
        "total_annotations": total_annotations,
        "total_predictions": total_predictions,
        "matches": matches,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "iou_threshold": iou_threshold,
    }


@app.get("/statistics")
async def get_system_statistics():
    """Get system-wide statistics"""
    stats = storage_manager.get_storage_stats()

    # Add more detailed statistics
    images = storage_manager.list_images()
    total_annotations = sum(img.annotation_count for img in images)
    images_with_predictions = sum(1 for img in images if img.has_ai_predictions)

    return {
        **stats,
        "total_annotations": total_annotations,
        "images_with_predictions": images_with_predictions,
        "average_annotations_per_image": (
            round(total_annotations / len(images), 2) if images else 0
        ),
    }


# Note: App is started by uvicorn command in Docker, not directly here
