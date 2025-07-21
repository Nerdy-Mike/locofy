import logging
import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.annotation_models import (
    Annotation,
    AnnotationCreate,
    AnnotationUpdate,
    ImageMetadata,
    LLMPrediction,
)
from services.llm_service import LLMUIDetectionService
from utils.file_storage import DuplicateImageError, FileStorageError, FileStorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        llm_service = LLMUIDetectionService(openai_api_key)
        logger.info("LLM service initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM service: {e}")
else:
    logger.warning("OpenAI API key not provided - LLM features will be disabled")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "UI Component Labeling API",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "image_upload": True,
            "annotation_management": True,
            "llm_predictions": llm_service is not None,
        },
        "endpoints": {
            "images": "/images",
            "annotations": "/annotations",
            "predictions": "/predictions",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check storage system
        storage_stats = storage_manager.get_storage_stats()

        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # This would be actual timestamp
            "services": {
                "storage": "operational",
                "llm": "operational" if llm_service else "disabled",
            },
            "storage_stats": (
                storage_stats
                if "error" not in storage_stats
                else {"error": "unavailable"}
            ),
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)},
        )


# Image Management Endpoints


@app.post("/images/upload", response_model=ImageMetadata)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a new image for labeling with comprehensive validation

    This endpoint implements the Phase 1.1 upload flow from DATAFLOW.md:
    - Validates file type, size, and content
    - Checks for duplicate images
    - Performs comprehensive image validation
    - Returns detailed metadata with validation info
    """
    logger.info(
        f"Upload request: filename='{file.filename}', content_type='{file.content_type}', size={file.size}"
    )

    try:
        # Basic file validation
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_filename",
                    "message": "Filename is required",
                    "field": "filename",
                },
            )

        # Validate content type if provided
        if file.content_type:
            allowed_types = [
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/gif",
                "image/bmp",
            ]
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "unsupported_file_type",
                        "message": f"Unsupported file type: {file.content_type}",
                        "allowed_types": allowed_types,
                        "provided_type": file.content_type,
                    },
                )

        # Read and validate file content
        try:
            content = await file.read()
        except Exception as e:
            logger.error(f"Failed to read uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "file_read_error",
                    "message": "Could not read uploaded file",
                    "details": str(e),
                },
            )

        # Check file size constraints
        max_size = 10 * 1024 * 1024  # 10MB
        min_size = 1024  # 1KB

        if len(content) < min_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "file_too_small",
                    "message": f"File too small (minimum {min_size} bytes)",
                    "actual_size": len(content),
                    "minimum_size": min_size,
                },
            )

        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "file_too_large",
                    "message": f"File too large (maximum {max_size / (1024*1024):.1f}MB)",
                    "actual_size_mb": round(len(content) / (1024 * 1024), 2),
                    "maximum_size_mb": round(max_size / (1024 * 1024), 1),
                },
            )

        # Reset file pointer for storage manager
        await file.seek(0)

        # Save image using enhanced storage manager
        try:
            metadata = storage_manager.save_image(content, file.filename, file)

            logger.info(
                f"Successfully uploaded image {metadata.id}: {metadata.filename} ({metadata.width}x{metadata.height})"
            )

            return metadata

        except DuplicateImageError as e:
            logger.warning(f"Duplicate image upload attempt: {e}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "duplicate_image",
                    "message": "This image has already been uploaded",
                    "details": str(e),
                },
            )

        except FileStorageError as e:
            logger.error(f"Storage error during upload: {e}")

            # Parse validation errors for better user feedback
            if "Validation failed:" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "validation_failed",
                        "message": "Image validation failed",
                        "details": str(e).replace("Validation failed: ", ""),
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "storage_error",
                        "message": "Failed to save image",
                        "details": str(e),
                    },
                )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred during upload",
                "details": (
                    str(e)
                    if os.getenv("ENVIRONMENT") == "development"
                    else "Internal error"
                ),
            },
        )


@app.get("/images", response_model=List[ImageMetadata])
async def list_images():
    """List all uploaded images with enhanced error handling"""
    try:
        images = storage_manager.list_images()
        logger.info(f"Listed {len(images)} images")
        return images

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "list_error",
                "message": "Could not retrieve image list",
                "details": str(e),
            },
        )


@app.get("/images/{image_id}", response_model=ImageMetadata)
async def get_image_metadata(image_id: str):
    """Get metadata for a specific image with validation"""
    try:
        metadata = storage_manager.get_image_metadata(image_id)

        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "image_not_found",
                    "message": f"Image with ID '{image_id}' not found",
                    "image_id": image_id,
                },
            )

        return metadata

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error retrieving image metadata {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "metadata_error",
                "message": "Could not retrieve image metadata",
                "details": str(e),
            },
        )


@app.get("/images/{image_id}/file")
async def get_image_file(image_id: str):
    """Get the actual image file with enhanced validation"""
    try:
        image_path = storage_manager.get_image_path(image_id)

        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "image_file_not_found",
                    "message": f"Image file not found for ID '{image_id}'",
                    "image_id": image_id,
                },
            )

        # Get metadata for proper content type
        metadata = storage_manager.get_image_metadata(image_id)
        if metadata:
            # Map format to MIME type
            format_to_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "GIF": "image/gif",
                "BMP": "image/bmp",
            }
            media_type = format_to_mime.get(metadata.format, "image/jpeg")
            filename = metadata.filename
        else:
            media_type = "image/jpeg"
            filename = f"{image_id}.jpg"

        return FileResponse(
            path=str(image_path), media_type=media_type, filename=filename
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error serving image file {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "file_serve_error",
                "message": "Could not serve image file",
                "details": str(e),
            },
        )


@app.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """Delete an image and all associated data with enhanced validation"""
    try:
        # Check if image exists first
        metadata = storage_manager.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "image_not_found",
                    "message": f"Image with ID '{image_id}' not found",
                    "image_id": image_id,
                },
            )

        # Attempt deletion
        success = storage_manager.delete_image(image_id)

        if success:
            logger.info(f"Successfully deleted image {image_id}")
            return {
                "success": True,
                "message": f"Image '{image_id}' deleted successfully",
                "image_id": image_id,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "deletion_failed",
                    "message": f"Failed to delete image '{image_id}'",
                    "image_id": image_id,
                },
            )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error deleting image {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "deletion_error",
                "message": "Could not delete image",
                "details": str(e),
            },
        )


# Annotation Management Endpoints


@app.post("/images/{image_id}/annotations", response_model=Annotation)
async def create_annotation(image_id: str, annotation: AnnotationCreate):
    """Create a new annotation for an image with enhanced validation"""
    try:
        # Verify image exists
        metadata = storage_manager.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "image_not_found",
                    "message": f"Image with ID '{image_id}' not found",
                    "image_id": image_id,
                },
            )

        # Get existing annotations
        existing_annotations = storage_manager.get_annotations(image_id)

        # Create new annotation
        new_annotation = Annotation(
            image_id=image_id,
            bounding_box=annotation.bounding_box,
            tag=annotation.tag,
            annotator=annotation.annotator,
            confidence=annotation.confidence,
        )

        # Add to existing annotations
        existing_annotations.append(new_annotation)

        # Save all annotations
        storage_manager.save_annotations(image_id, existing_annotations)

        logger.info(f"Created annotation {new_annotation.id} for image {image_id}")
        return new_annotation

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error creating annotation for image {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "annotation_creation_error",
                "message": "Could not create annotation",
                "details": str(e),
            },
        )


@app.get("/images/{image_id}/annotations", response_model=List[Annotation])
async def get_annotations(image_id: str):
    """Get all annotations for an image"""
    try:
        # Verify image exists
        if not storage_manager.get_image_metadata(image_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "image_not_found",
                    "message": f"Image with ID '{image_id}' not found",
                    "image_id": image_id,
                },
            )

        annotations = storage_manager.get_annotations(image_id)
        return annotations

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error retrieving annotations for image {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "annotation_retrieval_error",
                "message": "Could not retrieve annotations",
                "details": str(e),
            },
        )


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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llm_service_unavailable",
                "message": "LLM service not available. Check OPENAI_API_KEY configuration.",
                "suggestion": "Ensure OPENAI_API_KEY is set in environment variables",
            },
        )

    try:
        # Verify image exists
        image_path = storage_manager.get_image_path(image_id)
        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "image_not_found",
                    "message": f"Image file not found for ID '{image_id}'",
                    "image_id": image_id,
                },
            )

        # Generate predictions
        predictions = llm_service.detect_ui_components(image_id, str(image_path))

        # Save predictions
        storage_manager.save_llm_predictions(predictions)

        logger.info(
            f"Generated {len(predictions.predictions)} predictions for image {image_id}"
        )
        return predictions

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error generating predictions for image {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "prediction_error",
                "message": "Could not generate predictions",
                "details": str(e),
            },
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


# System Management Endpoints


@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        storage_stats = storage_manager.get_storage_stats()

        system_stats = {
            "storage": storage_stats,
            "services": {
                "llm_enabled": llm_service is not None,
                "api_version": "1.0.0",
            },
            "environment": os.getenv("ENVIRONMENT", "production"),
        }

        return system_stats

    except Exception as e:
        logger.error(f"Error retrieving system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "stats_error",
                "message": "Could not retrieve system statistics",
                "details": str(e),
            },
        )


# Note: App is started by uvicorn command in Docker, not directly here
