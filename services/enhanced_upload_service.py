import asyncio
import logging
from typing import Optional

from fastapi import HTTPException, UploadFile

from models.annotation_models import ImageMetadata
from models.validation_models import (
    UIValidationRequest,
    ValidationResult,
    ValidationStatus,
)
from services.llm_service import LLMUIDetectionService
from services.ui_validation_service import UIImageValidationService
from utils.config import AppConfig
from utils.file_storage import DuplicateImageError, FileStorageError, FileStorageManager

logger = logging.getLogger(__name__)


class EnhancedUploadService:
    """Enhanced upload service with LLM validation capabilities"""

    def __init__(self, config: AppConfig):
        """
        Initialize the enhanced upload service

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize core services
        self.storage_manager = FileStorageManager(
            base_data_dir=config.data_directory, temp_dir=config.temp_directory
        )

        # Initialize LLM services
        self.llm_service = LLMUIDetectionService(api_key=config.openai_api_key)
        self.validation_service = UIImageValidationService(
            llm_service=self.llm_service, config=config.get_validation_config()
        )

        logger.info("EnhancedUploadService initialized")

    async def upload_image_with_validation(
        self,
        file: UploadFile,
        custom_validation_request: Optional[UIValidationRequest] = None,
    ) -> ImageMetadata:
        """
        Upload image with LLM validation as specified in DATAFLOW.md section 1.2

        This implements the enhanced upload flow:
        1. Basic file validation
        2. Save to temporary storage
        3. LLM validation
        4. Move to permanent storage (if valid) or cleanup (if invalid)

        Args:
            file: Uploaded file from FastAPI
            custom_validation_request: Optional custom validation parameters

        Returns:
            ImageMetadata: Complete metadata with validation info

        Raises:
            HTTPException: With appropriate error codes and messages
        """
        temp_info = None

        try:
            # Step 1: Basic file validation
            await self._validate_upload_file(file)

            # Step 2: Read file content
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="File content is empty")

            # Step 3: Save to temporary storage
            temp_info = self.storage_manager.save_temporary_file(
                file_content=file_content,
                filename=file.filename or "unknown.jpg",
                content_type=file.content_type or "application/octet-stream",
            )

            logger.info(f"File saved to temporary storage: {temp_info.temp_path}")

            # Step 4: LLM Validation
            validation_request = custom_validation_request or UIValidationRequest(
                image_path=temp_info.temp_path,
                confidence_threshold=self.config.llm_validation_confidence_threshold,
                timeout_seconds=self.config.llm_validation_timeout,
            )

            validation_result = await self.validation_service.validate_web_ui_image(
                image_path=temp_info.temp_path, request=validation_request
            )

            logger.info(
                f"LLM validation completed: valid={validation_result.valid}, "
                f"confidence={validation_result.confidence:.3f}, "
                f"reason='{validation_result.reason}'"
            )

            # Step 5: Handle validation result
            if not validation_result.valid:
                # Validation failed - cleanup temp file and reject upload
                self.storage_manager.cleanup_temporary_file(temp_info)

                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Image rejected: Not a valid UI screenshot",
                        "reason": validation_result.reason,
                        "confidence": validation_result.confidence,
                        "status": validation_result.status.value,
                    },
                )

            # Step 6: Save validated image to permanent storage
            metadata = self.storage_manager.save_validated_image(
                temp_info=temp_info, validation_result=validation_result
            )

            logger.info(f"Image successfully uploaded and validated: {metadata.id}")
            return metadata

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise

        except DuplicateImageError as e:
            # Cleanup temp file and return duplicate error
            if temp_info:
                self.storage_manager.cleanup_temporary_file(temp_info)

            raise HTTPException(
                status_code=409, detail=f"Duplicate image detected: {str(e)}"
            )

        except FileStorageError as e:
            # Cleanup temp file and return storage error
            if temp_info:
                self.storage_manager.cleanup_temporary_file(temp_info)

            raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

        except Exception as e:
            # Cleanup temp file and return generic error
            if temp_info:
                self.storage_manager.cleanup_temporary_file(temp_info)

            logger.error(f"Unexpected error during upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    async def _validate_upload_file(self, file: UploadFile):
        """
        Perform basic file validation before processing

        Args:
            file: Uploaded file to validate

        Raises:
            HTTPException: If validation fails
        """
        # Check filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check content type
        allowed_types = self.config.get_allowed_file_types_list()
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. "
                f"Allowed types: {', '.join(allowed_types)}",
            )

        # Check file size
        if file.size and file.size > (self.config.max_file_size_mb * 1024 * 1024):
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {self.config.max_file_size_mb}MB",
            )

    def get_validation_stats(self) -> dict:
        """Get validation and storage statistics"""
        return {
            "validation_cache": self.validation_service.get_cache_stats(),
            "temp_files": self.storage_manager.get_temp_files_stats(),
            "storage": self.storage_manager.get_storage_stats(),
        }

    def cleanup_expired_files(self):
        """Clean up expired temporary files"""
        self.storage_manager.cleanup_expired_temp_files()

    def clear_validation_cache(self):
        """Clear the validation cache"""
        self.validation_service.clear_validation_cache()


class UploadServiceManager:
    """Manager for upload service lifecycle"""

    _instance: Optional[EnhancedUploadService] = None
    _cleanup_task_started: bool = False

    @classmethod
    def get_service(cls, config: Optional[AppConfig] = None) -> EnhancedUploadService:
        """Get or create upload service instance"""
        if cls._instance is None:
            if config is None:
                from utils.config import get_config

                config = get_config()

            cls._instance = EnhancedUploadService(config)

            # Start background cleanup task only if we're in an async context
            cls._start_cleanup_task_if_possible()

        return cls._instance

    @classmethod
    def _start_cleanup_task_if_possible(cls):
        """Start cleanup task only if we're in an async context"""
        if cls._cleanup_task_started:
            return

        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                asyncio.create_task(cls._cleanup_task())
                cls._cleanup_task_started = True
        except RuntimeError:
            # No event loop running, cleanup task will be started when needed
            pass

    @classmethod
    async def start_background_tasks(cls):
        """Manually start background tasks (call this from FastAPI startup)"""
        if not cls._cleanup_task_started:
            asyncio.create_task(cls._cleanup_task())
            cls._cleanup_task_started = True

    @classmethod
    async def _cleanup_task(cls):
        """Background task for cleaning up expired files"""
        while True:
            try:
                if cls._instance:
                    cls._instance.cleanup_expired_files()

                # Sleep for cleanup interval
                if cls._instance:
                    cleanup_interval = cls._instance.config.temp_file_cleanup_interval
                else:
                    cleanup_interval = 3600  # Default 1 hour

                await asyncio.sleep(cleanup_interval)

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error


# Convenience function for getting service
def get_upload_service() -> EnhancedUploadService:
    """Get the upload service instance"""
    return UploadServiceManager.get_service()
