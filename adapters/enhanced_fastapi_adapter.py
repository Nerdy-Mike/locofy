"""
Enhanced FastAPI adapter demonstrating integration with LLM validation service

This shows how to integrate the new enhanced upload flow (DATAFLOW.md section 1.2)
into your existing FastAPI application.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Path, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response

from models.annotation_models import (
    Annotation,
    AnnotationConfig,
    AnnotationStatus,
    BatchAnnotationRequest,
    BatchAnnotationResponse,
    ImageMetadata,
    ValidationResult,
)
from models.validation_models import UIValidationRequest
from services.annotation_validation_service import AnnotationValidationService
from services.enhanced_upload_service import EnhancedUploadService, get_upload_service
from services.quality_metrics_service import QualityMetricsService
from utils.config import AppConfig, get_config, validate_environment
from utils.file_storage import FileStorageManager

# from .fastapi_adapter import FastAPIAdapter  # Not needed for standalone implementation

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Locofy UI Component Labeling System",
    description="Enhanced image upload with LLM validation",
    version="1.2.0",
)


@app.on_event("startup")
async def startup_event():
    """Application startup - validate environment and setup logging"""
    try:
        # Load and validate configuration
        config = get_config()
        config.setup_logging()
        validate_environment()

        # Initialize services (this will be done automatically on first request)
        upload_service = get_upload_service()

        # Start background tasks
        from services.enhanced_upload_service import UploadServiceManager

        await UploadServiceManager.start_background_tasks()

        logger.info("🚀 Enhanced upload service started successfully")
        logger.info(f"🔧 LLM validation enabled: {config.llm_validation_enabled}")
        logger.info(f"📁 Data directory: {config.data_directory}")

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise


def get_upload_service_dependency() -> EnhancedUploadService:
    """Dependency injection for upload service"""
    return get_upload_service()


@app.post("/images/upload", response_model=ImageMetadata)
async def upload_image_with_validation(
    file: UploadFile = File(...),
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> ImageMetadata:
    """
    Upload image with LLM validation (Enhanced Flow 1.2)

    This endpoint implements the complete enhanced upload flow:
    1. Basic file validation (type, size)
    2. Save to temporary storage
    3. LLM validation to ensure it's a UI screenshot
    4. Move to permanent storage (if valid) or reject (if invalid)

    Args:
        file: Image file to upload
        upload_service: Enhanced upload service (injected)

    Returns:
        ImageMetadata: Complete metadata with validation results

    Raises:
        HTTPException 400: File validation failed or LLM rejected image
        HTTPException 409: Duplicate image detected
        HTTPException 500: Server error during processing
    """
    try:
        result = await upload_service.upload_image_with_validation(file)

        # Log successful upload
        logger.info(
            f"✅ Image uploaded successfully: {result.id} "
            f"(confidence: {result.llm_validation_result.confidence:.3f})"
        )

        return result

    except HTTPException:
        # Re-raise HTTP exceptions (they have proper status codes)
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"❌ Unexpected upload error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during upload"
        )


@app.post("/images/upload-custom", response_model=ImageMetadata)
async def upload_image_with_custom_validation(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = 0.7,
    timeout_seconds: Optional[int] = 10,
    include_element_detection: Optional[bool] = False,
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> ImageMetadata:
    """
    Upload image with custom validation parameters

    This allows fine-tuning of the validation process per request.

    Args:
        file: Image file to upload
        confidence_threshold: Minimum confidence required (0.0-1.0)
        timeout_seconds: Maximum time to wait for LLM (1-60 seconds)
        include_element_detection: Whether to include detailed UI element detection
        upload_service: Enhanced upload service (injected)

    Returns:
        ImageMetadata: Complete metadata with validation results
    """
    try:
        # Create custom validation request
        validation_request = UIValidationRequest(
            image_path="",  # Will be set by the service
            confidence_threshold=confidence_threshold or 0.7,
            timeout_seconds=timeout_seconds or 10,
            include_element_detection=include_element_detection or False,
        )

        result = await upload_service.upload_image_with_validation(
            file, custom_validation_request=validation_request
        )

        logger.info(f"✅ Custom validation upload successful: {result.id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Custom validation upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during custom validation upload",
        )


@app.get("/stats/validation")
async def get_validation_stats(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> dict:
    """
    Get validation and storage statistics

    Returns information about:
    - Validation cache usage
    - Temporary files status
    - Storage statistics
    """
    try:
        stats = upload_service.get_validation_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error retrieving validation stats: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve validation statistics"
        )


@app.post("/admin/cleanup-temp-files")
async def cleanup_temporary_files(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> dict:
    """
    Manually trigger cleanup of expired temporary files

    This is useful for admin operations or debugging.
    """
    try:
        upload_service.cleanup_expired_files()
        return {"status": "success", "message": "Temporary file cleanup completed"}
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup temporary files")


@app.post("/admin/clear-validation-cache")
async def clear_validation_cache(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> dict:
    """
    Clear the LLM validation cache

    This forces fresh validation for all subsequent uploads.
    """
    try:
        upload_service.clear_validation_cache()
        return {"status": "success", "message": "Validation cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing validation cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear validation cache")


# === MISSING ENDPOINTS FOR FRONTEND COMPATIBILITY ===


@app.get("/images", response_model=List[ImageMetadata])
async def list_images(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> List[ImageMetadata]:
    """
    Get list of all uploaded images

    This endpoint is expected by the Streamlit frontend for the image gallery.
    """
    try:
        # Get images from storage manager
        images = upload_service.storage_manager.list_images()
        return images
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image list")


@app.get("/images/{image_id}", response_model=ImageMetadata)
async def get_image_metadata(
    image_id: str = Path(..., description="ID of the image"),
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> ImageMetadata:
    """
    Get metadata for a specific image

    This endpoint is expected by the Streamlit frontend.
    """
    try:
        metadata = upload_service.storage_manager.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image metadata {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image metadata")


@app.get("/images/{image_id}/file")
async def get_image_file(
    image_id: str = Path(..., description="ID of the image"),
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
):
    """
    Get the actual image file content

    This endpoint is expected by the Streamlit frontend for displaying images.
    """
    try:
        # Get image path
        image_path = upload_service.storage_manager.get_image_path(image_id)
        if not image_path or not image_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Image file for ID {image_id} not found"
            )

        # Read and return file content
        with open(image_path, "rb") as f:
            image_content = f.read()

        # Determine content type based on file extension
        extension = image_path.suffix.lower()
        content_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }
        content_type = content_type_map.get(extension, "image/jpeg")

        return Response(
            content=image_content,
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename={image_path.name}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image file {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image file")


@app.delete("/images/{image_id}")
async def delete_image(
    image_id: str = Path(..., description="ID of the image to delete"),
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> dict:
    """
    Delete an image and all associated data

    This endpoint is expected by the Streamlit frontend.
    """
    try:
        # Check if image exists
        metadata = upload_service.storage_manager.get_image_metadata(image_id)
        if not metadata:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        # Delete the image
        success = upload_service.storage_manager.delete_image(image_id)

        if success:
            logger.info(f"Successfully deleted image: {image_id}")
            return {
                "message": f"Image {image_id} deleted successfully",
                "deleted_image_id": image_id,
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to delete image completely"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete image")


@app.get("/statistics")
async def get_statistics(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> dict:
    """
    Get system-wide statistics

    This endpoint is expected by the Streamlit frontend.
    """
    try:
        # Get storage statistics
        storage_stats = upload_service.storage_manager.get_storage_stats()

        # Get validation statistics
        validation_stats = upload_service.get_validation_stats()

        # Combine statistics
        return {
            "storage": storage_stats,
            "validation": validation_stats,
            "system": {
                "llm_validation_enabled": upload_service.config.llm_validation_enabled,
                "confidence_threshold": upload_service.config.llm_validation_confidence_threshold,
                "max_file_size_mb": upload_service.config.max_file_size_mb,
            },
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


# === ANNOTATION ENDPOINTS ===

# Initialize annotation services globally
_annotation_config = AnnotationConfig()


def get_annotation_validator(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> AnnotationValidationService:
    """Dependency injection for annotation validation service"""
    return AnnotationValidationService(
        storage_manager=upload_service.storage_manager, config=_annotation_config
    )


def get_quality_metrics_service(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
) -> QualityMetricsService:
    """Dependency injection for quality metrics service"""
    return QualityMetricsService(storage_manager=upload_service.storage_manager)


@app.get("/annotations/{image_id}", response_model=List[Annotation])
async def get_annotations(
    image_id: str = Path(..., description="ID of the image"),
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
):
    """
    Get all annotations for a specific image

    Args:
        image_id: ID of the image to get annotations for

    Returns:
        List[Annotation]: All annotations for the image

    Raises:
        HTTPException: If image not found or error loading annotations
    """
    try:
        # Verify image exists
        image_metadata = upload_service.storage_manager.get_image_metadata(image_id)
        if not image_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {image_id}",
            )

        # Load annotations
        annotations = upload_service.storage_manager.get_annotations(image_id)

        logger.info(f"Retrieved {len(annotations)} annotations for image {image_id}")
        return annotations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading annotations for image {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading annotations: {str(e)}",
        )


@app.post("/annotations/batch", response_model=BatchAnnotationResponse)
async def save_annotation_batch(
    request: BatchAnnotationRequest,
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
    annotation_validator: AnnotationValidationService = Depends(
        get_annotation_validator
    ),
    quality_service: QualityMetricsService = Depends(get_quality_metrics_service),
):
    """
    Save a batch of annotations for an image

    Args:
        request: BatchAnnotationRequest containing image_id, created_by, and annotations

    Returns:
        BatchAnnotationResponse: Results of the batch save operation

    Raises:
        HTTPException: If validation fails or save operation fails
    """
    start_time = time.time()

    try:
        # Verify image exists
        image_metadata = upload_service.storage_manager.get_image_metadata(
            request.image_id
        )
        if not image_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {request.image_id}",
            )

        # Validate annotation batch
        validation_result = annotation_validator.validate_annotation_batch(
            annotations=request.annotations,
            image_id=request.image_id,
            created_by=request.created_by,
        )

        # If validation failed, return errors
        if not validation_result.valid:
            error_details = {
                "message": "Annotation validation failed",
                "errors": [
                    {"field": err.field, "message": err.message, "value": err.value}
                    for err in validation_result.errors
                ],
            }
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_details
            )

        # Convert requests to full annotation objects
        annotations_to_save = []
        for i, annotation_request in enumerate(request.annotations):
            # Check if this specific annotation has conflicts
            annotation_conflicts = [
                conflict
                for conflict in validation_result.conflicts
                if conflict.annotation_id == f"temp_{i}"
            ]

            annotation = Annotation(
                image_id=request.image_id,
                bounding_box=annotation_request.bounding_box,
                tag=annotation_request.tag,
                confidence=annotation_request.confidence,
                created_by=request.created_by,
                status=(
                    AnnotationStatus.CONFLICTED
                    if annotation_conflicts
                    else AnnotationStatus.ACTIVE
                ),
                reasoning=annotation_request.reasoning,
            )

            # Set conflicts if any found
            if annotation_conflicts:
                annotation.conflicts_with = []
                for conflict in annotation_conflicts:
                    annotation.conflicts_with.extend(conflict.conflicts_with)

            annotations_to_save.append(annotation)

        # Save annotation batch
        success = upload_service.storage_manager.save_annotation_batch(
            annotations_to_save
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save annotation batch",
            )

        # Update quality metrics after successful save
        quality_metrics = quality_service.update_image_quality_metrics(request.image_id)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare response
        response = BatchAnnotationResponse(
            saved_count=len(annotations_to_save),
            annotation_ids=[ann.id for ann in annotations_to_save],
            conflicts=validation_result.conflicts,
            warnings=[warn.message for warn in validation_result.warnings],
            processing_time=processing_time,
        )

        logger.info(
            f"Successfully saved batch of {len(annotations_to_save)} annotations "
            f"for image {request.image_id} in {processing_time:.2f}s"
            f"({len(validation_result.conflicts)} conflicts detected)"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving annotation batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving annotations: {str(e)}",
        )


@app.get("/annotations/{image_id}/conflicts")
async def get_annotation_conflicts(
    image_id: str = Path(..., description="ID of the image"),
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
):
    """
    Get conflicts for annotations on a specific image

    Args:
        image_id: ID of the image to check for conflicts

    Returns:
        Dict: Conflict information for the image
    """
    try:
        # Verify image exists
        image_metadata = upload_service.storage_manager.get_image_metadata(image_id)
        if not image_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {image_id}",
            )

        # Load annotations
        annotations = upload_service.storage_manager.get_annotations(image_id)

        # Filter for conflicted annotations
        conflicted_annotations = [
            ann for ann in annotations if ann.status == AnnotationStatus.CONFLICTED
        ]

        return {
            "image_id": image_id,
            "total_annotations": len(annotations),
            "conflicted_annotations": len(conflicted_annotations),
            "conflicts": [
                {
                    "annotation_id": ann.id,
                    "conflicts_with": ann.conflicts_with,
                    "tag": ann.tag,
                    "created_by": ann.created_by,
                    "created_at": ann.created_at.isoformat(),
                    "bounding_box": ann.bounding_box.dict(),
                }
                for ann in conflicted_annotations
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conflicts for image {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conflicts: {str(e)}",
        )


@app.get("/annotations/statistics")
async def get_annotation_statistics(
    upload_service: EnhancedUploadService = Depends(get_upload_service_dependency),
):
    """
    Get comprehensive statistics about annotations

    Returns:
        Dict: Statistics about annotations across all images
    """
    try:
        stats = upload_service.storage_manager.get_annotation_statistics()

        # Add validation configuration info
        stats["configuration"] = {
            "min_box_width": _annotation_config.min_box_width,
            "min_box_height": _annotation_config.min_box_height,
            "overlap_threshold": _annotation_config.overlap_threshold,
            "max_annotations_per_batch": _annotation_config.max_annotations_per_batch,
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting annotation statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting statistics: {str(e)}",
        )


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information"""
    config = get_config()
    return {
        "message": "🚀 Locofy Enhanced Upload Service with LLM Validation & Annotations",
        "version": "1.2.0",
        "status": "operational",
        "llm_validation_enabled": config.llm_validation_enabled,
        "endpoints": {
            "health": "/health",
            "upload": "/images/upload",
            "upload_custom": "/images/upload-custom",
            "validation_stats": "/stats/validation",
            "annotations": "/annotations/{image_id}",
            "annotation_batch": "/annotations/batch",
            "annotation_conflicts": "/annotations/{image_id}/conflicts",
            "annotation_stats": "/annotations/statistics",
            "frontend": "http://localhost:8501",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint"""
    config = get_config()
    return {
        "status": "healthy",
        "version": "1.2.0",
        "llm_validation_enabled": config.llm_validation_enabled,
        "data_directory": config.data_directory,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler with enhanced error details"""

    # Log non-client errors
    if exc.status_code >= 500:
        logger.error(f"Server error {exc.status_code}: {exc.detail}")
    elif exc.status_code >= 400:
        logger.warning(f"Client error {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": logger.time.time() if hasattr(logger, "time") else None,
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Get configuration
    config = get_config()

    # Run the application
    uvicorn.run(
        "enhanced_fastapi_adapter:app",
        host=config.api_host,
        port=config.api_port,
        debug=config.api_debug,
        reload=config.api_debug,
    )
