"""
Enhanced FastAPI adapter demonstrating integration with LLM validation service

This shows how to integrate the new enhanced upload flow (DATAFLOW.md section 1.2)
into your existing FastAPI application.
"""

import logging
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Path, UploadFile
from fastapi.responses import JSONResponse, Response

from models.annotation_models import ImageMetadata
from models.validation_models import UIValidationRequest
from services.enhanced_upload_service import EnhancedUploadService, get_upload_service
from utils.config import AppConfig, get_config, validate_environment

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


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information"""
    config = get_config()
    return {
        "message": "🚀 Locofy Enhanced Upload Service with LLM Validation",
        "version": "1.2.0",
        "status": "operational",
        "llm_validation_enabled": config.llm_validation_enabled,
        "endpoints": {
            "health": "/health",
            "upload": "/images/upload",
            "upload_custom": "/images/upload-custom",
            "validation_stats": "/stats/validation",
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
