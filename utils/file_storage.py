import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from PIL import Image

from models.annotation_models import (
    Annotation,
    ImageMetadata,
    ImageValidationInfo,
    LLMPrediction,
)
from models.validation_models import TemporaryFileInfo
from utils.validation import (
    ImageValidationResult,
    InputSanitizer,
    UploadValidator,
    ValidationResult,
)

# Configure logging
logger = logging.getLogger(__name__)


class FileStorageError(Exception):
    """Custom exception for file storage operations"""

    pass


class DuplicateImageError(FileStorageError):
    """Exception raised when duplicate image is detected"""

    pass


class FileStorageManager:
    """Enhanced file storage manager with comprehensive validation and error handling"""

    def __init__(self, base_data_dir: str = "data", temp_dir: Optional[str] = None):
        self.base_dir = Path(base_data_dir)
        self.images_dir = self.base_dir / "images"
        self.annotations_dir = self.base_dir / "annotations"
        self.metadata_dir = self.base_dir / "metadata"
        self.predictions_dir = self.base_dir / "predictions"
        
        # Set up temporary directory for upload processing
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "locofy_uploads"
        
        # Create directories if they don't exist
        self._create_directories()

        # Track known checksums for duplicate detection
        self._checksum_cache: Dict[str, str] = {}
        self._load_checksum_cache()
        
        # Track temporary files for cleanup
        self._temp_files: Dict[str, TemporaryFileInfo] = {}

    def _create_directories(self):
        """Create necessary directories for file storage"""
        for directory in [
            self.images_dir,
            self.annotations_dir,
            self.metadata_dir,
            self.predictions_dir,
            self.temp_dir,
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created/verified directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise FileStorageError(f"Cannot create storage directory: {directory}")

    def _load_checksum_cache(self):
        """Load existing image checksums for duplicate detection"""
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, "r") as f:
                        data = json.load(f)

                    if "validation_info" in data and data["validation_info"]:
                        checksum = data["validation_info"]["checksum"]
                        image_id = data["id"]
                        self._checksum_cache[checksum] = image_id

                except Exception as e:
                    logger.warning(f"Could not load checksum from {metadata_file}: {e}")

        except Exception as e:
            logger.warning(f"Could not load checksum cache: {e}")

    def save_image(
        self, file_content: bytes, filename: str, original_file=None
    ) -> ImageMetadata:
        """
        Save uploaded image with comprehensive validation

        Args:
            file_content: Raw file bytes content
            filename: Original filename
            original_file: Optional original file object for additional metadata

        Returns:
            ImageMetadata: Complete metadata with validation info

        Raises:
            FileStorageError: If saving fails
            DuplicateImageError: If image already exists
        """
        try:
            # We now receive file_content directly, no need to read from file object
            if not file_content:
                raise FileStorageError("File content is empty")

            # Get content type from original file if available
            content_type = (
                getattr(original_file, "content_type", None)
                or "application/octet-stream"
            )

            # Comprehensive validation
            validation_result = UploadValidator.validate_upload_file(
                file_content, filename, content_type
            )

            if not validation_result.valid:
                error_details = "; ".join(
                    [f"{err.field}: {err.message}" for err in validation_result.errors]
                )
                raise FileStorageError(f"Validation failed: {error_details}")

            # Calculate checksum for duplicate detection
            checksum = hashlib.md5(file_content).hexdigest()

            # Check for duplicates
            if checksum in self._checksum_cache:
                existing_id = self._checksum_cache[checksum]
                raise DuplicateImageError(
                    f"Image already exists with ID: {existing_id}"
                )

            # Generate unique image ID
            image_id = str(uuid4())

            # Sanitize filename
            sanitized_filename = UploadValidator.sanitize_filename(filename)

            # Determine file extension
            file_ext = Path(filename).suffix.lower()
            if not file_ext:
                # Try to determine from PIL
                try:
                    temp_image = Image.open(BytesIO(file_content))
                    format_to_ext = {
                        "JPEG": ".jpg",
                        "PNG": ".png",
                        "GIF": ".gif",
                        "BMP": ".bmp",
                    }
                    file_ext = format_to_ext.get(temp_image.format, ".jpg")
                except:
                    file_ext = ".jpg"  # default

            # Create storage filename
            storage_filename = f"{image_id}{file_ext}"
            storage_path = self.images_dir / storage_filename

            # Save the image file
            try:
                with open(storage_path, "wb") as f:
                    f.write(file_content)
                logger.info(f"Saved image file: {storage_path}")
            except Exception as e:
                raise FileStorageError(f"Failed to save image file: {e}")

            # Extract image metadata using PIL
            try:
                with Image.open(storage_path) as img:
                    width, height = img.size
                    format_name = img.format or "UNKNOWN"
            except Exception as e:
                # Clean up the saved file if PIL fails
                try:
                    os.unlink(storage_path)
                except:
                    pass
                raise FileStorageError(
                    f"Invalid image file - PIL processing failed: {e}"
                )

            # Create validation info
            validation_info = ImageValidationInfo(
                checksum=checksum,
                original_filename=filename,
                sanitized_filename=sanitized_filename,
                file_size_bytes=len(file_content),
            )

            # Create metadata
            metadata = ImageMetadata(
                id=image_id,
                filename=sanitized_filename,
                file_path=str(storage_path),
                file_size=len(file_content),
                width=width,
                height=height,
                format=format_name,
                validation_info=validation_info,
                processing_status="completed",
            )

            # Save metadata
            self.save_image_metadata(metadata)

            # Update checksum cache
            self._checksum_cache[checksum] = image_id

            logger.info(
                f"Successfully saved image {image_id} ({width}x{height}, {len(file_content)} bytes)"
            )
            return metadata

        except (FileStorageError, DuplicateImageError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving image: {e}")
            raise FileStorageError(f"Unexpected error during image save: {e}")

    def save_image_metadata(self, metadata: ImageMetadata):
        """Save image metadata to JSON file with error handling"""
        try:
            metadata_path = self.metadata_dir / f"{metadata.id}.json"

            # Convert to dict and handle datetime serialization
            metadata_dict = metadata.dict()

            # Ensure all datetime objects are properly serialized
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                return obj

            metadata_dict = serialize_datetime(metadata_dict)

            with open(metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved metadata: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.id}: {e}")
            raise FileStorageError(f"Cannot save image metadata: {e}")

    def get_image_metadata(self, image_id: str) -> Optional[ImageMetadata]:
        """Load image metadata from JSON file with error handling"""
        try:
            # Validate image_id format
            if not image_id or not isinstance(image_id, str):
                return None

            metadata_path = self.metadata_dir / f"{image_id}.json"
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found: {metadata_path}")
                return None

            with open(metadata_path, "r") as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            if "upload_time" in data and isinstance(data["upload_time"], str):
                data["upload_time"] = datetime.fromisoformat(
                    data["upload_time"].replace("Z", "+00:00")
                )

            if "validation_info" in data and data["validation_info"]:
                if "validation_timestamp" in data["validation_info"]:
                    data["validation_info"]["validation_timestamp"] = (
                        datetime.fromisoformat(
                            data["validation_info"]["validation_timestamp"].replace(
                                "Z", "+00:00"
                            )
                        )
                    )

            return ImageMetadata(**data)

        except Exception as e:
            logger.error(f"Failed to load metadata for {image_id}: {e}")
            return None

    def get_image_path(self, image_id: str) -> Optional[Path]:
        """Get the file path for an image with validation"""
        try:
            metadata = self.get_image_metadata(image_id)
            if metadata:
                image_path = Path(metadata.file_path)
                if image_path.exists():
                    return image_path
                else:
                    logger.error(f"Image file missing: {image_path}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting image path for {image_id}: {e}")
            return None

    def list_images(self) -> List[ImageMetadata]:
        """List all available images with error handling"""
        images = []

        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    metadata = self.get_image_metadata(metadata_file.stem)
                    if metadata:
                        images.append(metadata)
                except Exception as e:
                    logger.warning(f"Error loading metadata {metadata_file}: {e}")

        except Exception as e:
            logger.error(f"Error listing images: {e}")

        return sorted(images, key=lambda x: x.upload_time, reverse=True)

    def delete_image(self, image_id: str) -> bool:
        """
        Delete an image and all associated data

        Args:
            image_id: ID of image to delete

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            metadata = self.get_image_metadata(image_id)
            if not metadata:
                logger.warning(f"Cannot delete - image metadata not found: {image_id}")
                return False

            success = True

            # Delete image file
            try:
                image_path = Path(metadata.file_path)
                if image_path.exists():
                    os.unlink(image_path)
                    logger.info(f"Deleted image file: {image_path}")
            except Exception as e:
                logger.error(f"Failed to delete image file: {e}")
                success = False

            # Delete metadata file
            try:
                metadata_path = self.metadata_dir / f"{image_id}.json"
                if metadata_path.exists():
                    os.unlink(metadata_path)
                    logger.info(f"Deleted metadata file: {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to delete metadata file: {e}")
                success = False

            # Delete annotations if they exist
            try:
                annotations_path = self.annotations_dir / f"{image_id}.json"
                if annotations_path.exists():
                    os.unlink(annotations_path)
                    logger.info(f"Deleted annotations file: {annotations_path}")
            except Exception as e:
                logger.warning(f"Could not delete annotations file: {e}")

            # Delete predictions if they exist
            try:
                predictions_path = self.predictions_dir / f"{image_id}.json"
                if predictions_path.exists():
                    os.unlink(predictions_path)
                    logger.info(f"Deleted predictions file: {predictions_path}")
            except Exception as e:
                logger.warning(f"Could not delete predictions file: {e}")

            # Remove from checksum cache
            if (
                metadata.validation_info
                and metadata.validation_info.checksum in self._checksum_cache
            ):
                del self._checksum_cache[metadata.validation_info.checksum]

            return success

        except Exception as e:
            logger.error(f"Error deleting image {image_id}: {e}")
            return False

    def save_annotations(self, image_id: str, annotations: List[Annotation]):
        """Save annotations for an image with validation"""
        try:
            # Validate image exists
            if not self.get_image_metadata(image_id):
                raise FileStorageError(
                    f"Cannot save annotations - image not found: {image_id}"
                )

            annotations_path = self.annotations_dir / f"{image_id}.json"
            annotations_data = [ann.dict() for ann in annotations]

            # Serialize datetime objects
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                return obj

            annotations_data = serialize_datetime(annotations_data)

            with open(annotations_path, "w") as f:
                json.dump(annotations_data, f, indent=2, ensure_ascii=False)

            # Update annotation count in metadata
            metadata = self.get_image_metadata(image_id)
            if metadata:
                metadata.annotation_count = len(annotations)
                self.save_image_metadata(metadata)

            logger.info(f"Saved {len(annotations)} annotations for image {image_id}")

        except Exception as e:
            logger.error(f"Failed to save annotations for {image_id}: {e}")
            raise FileStorageError(f"Cannot save annotations: {e}")

    def get_annotations(self, image_id: str) -> List[Annotation]:
        """Load annotations for an image with error handling"""
        try:
            annotations_path = self.annotations_dir / f"{image_id}.json"

            if not annotations_path.exists():
                return []

            with open(annotations_path, "r") as f:
                data = json.load(f)

            annotations = []
            for ann_data in data:
                try:
                    # Convert datetime strings back to datetime objects
                    if "created_at" in ann_data and isinstance(
                        ann_data["created_at"], str
                    ):
                        ann_data["created_at"] = datetime.fromisoformat(
                            ann_data["created_at"].replace("Z", "+00:00")
                        )
                    if "updated_at" in ann_data and isinstance(
                        ann_data["updated_at"], str
                    ):
                        ann_data["updated_at"] = datetime.fromisoformat(
                            ann_data["updated_at"].replace("Z", "+00:00")
                        )
                    if "reviewed_at" in ann_data and isinstance(
                        ann_data["reviewed_at"], str
                    ):
                        ann_data["reviewed_at"] = datetime.fromisoformat(
                            ann_data["reviewed_at"].replace("Z", "+00:00")
                        )

                    annotation = Annotation(**ann_data)
                    annotations.append(annotation)
                except Exception as e:
                    logger.warning(f"Skipping invalid annotation data: {e}")

            return annotations

        except Exception as e:
            logger.error(f"Failed to load annotations for {image_id}: {e}")
            return []

    def load_annotations(self, image_id: str) -> List[Annotation]:
        """Alias for get_annotations for consistency with validation service"""
        return self.get_annotations(image_id)

    def save_annotation_batch(self, annotations: List[Annotation]) -> bool:
        """
        Save multiple annotations atomically for a single image
        
        Args:
            annotations: List of annotations to save (must all be for same image)
            
        Returns:
            bool: True if all annotations saved successfully
            
        Raises:
            FileStorageError: If batch save fails
        """
        if not annotations:
            return True
            
        # Verify all annotations are for the same image
        image_id = annotations[0].image_id
        if not all(ann.image_id == image_id for ann in annotations):
            raise FileStorageError("All annotations in batch must be for the same image")
        
        try:
            # Load existing annotations
            existing_annotations = self.get_annotations(image_id)
            
            # Combine existing and new annotations
            all_annotations = existing_annotations + annotations
            
            # Save all annotations atomically
            self.save_annotations(image_id, all_annotations)
            
            logger.info(f"Successfully saved batch of {len(annotations)} annotations for image {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save annotation batch for {image_id}: {e}")
            raise FileStorageError(f"Cannot save annotation batch: {e}")

    def update_annotation_status(self, image_id: str, annotation_id: str, 
                               status: str, reviewed_by: Optional[str] = None) -> bool:
        """
        Update the status of a specific annotation
        
        Args:
            image_id: ID of the image
            annotation_id: ID of the annotation to update
            status: New status value
            reviewed_by: Optional reviewer identifier
            
        Returns:
            bool: True if update successful
        """
        try:
            annotations = self.get_annotations(image_id)
            
            # Find and update the specific annotation
            updated = False
            for annotation in annotations:
                if annotation.id == annotation_id:
                    annotation.status = status
                    if reviewed_by:
                        annotation.reviewed_by = reviewed_by
                        annotation.reviewed_at = datetime.utcnow()
                    updated = True
                    break
            
            if not updated:
                logger.warning(f"Annotation {annotation_id} not found for image {image_id}")
                return False
            
            # Save updated annotations
            self.save_annotations(image_id, annotations)
            
            logger.info(f"Updated annotation {annotation_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update annotation status: {e}")
            return False

    def save_annotation_file_structure(self, image_id: str, annotations: List[Annotation]):
        """
        Save annotations in the enhanced file structure format
        
        Args:
            image_id: ID of the image
            annotations: List of annotations to save
        """
        try:
            annotations_path = self.annotations_dir / f"{image_id}.json"
            
            # Create the enhanced file structure as documented in DATAFLOW.md
            file_structure = {
                "image_id": image_id,
                "last_updated": datetime.utcnow().isoformat(),
                "annotation_count": len(annotations),
                "annotations": []
            }
            
            # Convert annotations to dict format
            for annotation in annotations:
                ann_dict = annotation.dict()
                
                # Serialize datetime objects
                def serialize_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: serialize_datetime(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_datetime(item) for item in obj]
                    return obj
                
                ann_dict = serialize_datetime(ann_dict)
                file_structure["annotations"].append(ann_dict)
            
            # Write to file
            with open(annotations_path, "w") as f:
                json.dump(file_structure, f, indent=2, ensure_ascii=False)
            
            # Update image metadata annotation count and quality metrics
            metadata = self.get_image_metadata(image_id)
            if metadata:
                metadata.annotation_count = len(annotations)
                # Check for conflicts
                metadata.has_conflicts = any(
                    ann.status == "conflicted" for ann in annotations
                )
                
                # Calculate simple quality score based on conflicts and annotation density
                if len(annotations) > 0:
                    conflict_ratio = sum(1 for ann in annotations if ann.status == "conflicted") / len(annotations)
                    # Simple quality score: fewer conflicts = higher quality
                    metadata.quality_score = max(0.0, 1.0 - conflict_ratio)
                else:
                    metadata.quality_score = None
                    
                self.save_image_metadata(metadata)
            
            logger.info(f"Saved {len(annotations)} annotations with enhanced structure for image {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to save annotation file structure for {image_id}: {e}")
            raise FileStorageError(f"Cannot save annotation file structure: {e}")

    def get_conflicted_annotations(self) -> List[Annotation]:
        """
        Get all annotations that have conflicts across all images
        
        Returns:
            List[Annotation]: All conflicted annotations
        """
        conflicted_annotations = []
        
        try:
            # Iterate through all annotation files
            for annotations_file in self.annotations_dir.glob("*.json"):
                try:
                    image_id = annotations_file.stem
                    annotations = self.get_annotations(image_id)
                    
                    # Filter for conflicted annotations
                    conflicted = [
                        ann for ann in annotations 
                        if ann.status == "conflicted"
                    ]
                    conflicted_annotations.extend(conflicted)
                    
                except Exception as e:
                    logger.warning(f"Error processing annotations file {annotations_file}: {e}")
            
            logger.info(f"Found {len(conflicted_annotations)} conflicted annotations")
            return conflicted_annotations
            
        except Exception as e:
            logger.error(f"Failed to get conflicted annotations: {e}")
            return []

    def get_annotation_statistics(self) -> Dict:
        """
        Get comprehensive statistics about annotations
        
        Returns:
            Dict: Statistics about annotations across all images
        """
        try:
            stats = {
                "total_annotations": 0,
                "annotations_by_status": {},
                "annotations_by_tag": {},
                "images_with_annotations": 0,
                "images_with_conflicts": 0,
                "average_annotations_per_image": 0
            }
            
            images_with_annotations = 0
            images_with_conflicts = 0
            
            # Process all annotation files
            for annotations_file in self.annotations_dir.glob("*.json"):
                try:
                    image_id = annotations_file.stem
                    annotations = self.get_annotations(image_id)
                    
                    if annotations:
                        images_with_annotations += 1
                        stats["total_annotations"] += len(annotations)
                        
                        # Check for conflicts
                        has_conflicts = any(ann.status == "conflicted" for ann in annotations)
                        if has_conflicts:
                            images_with_conflicts += 1
                        
                        # Count by status
                        for annotation in annotations:
                            status = annotation.status
                            stats["annotations_by_status"][status] = (
                                stats["annotations_by_status"].get(status, 0) + 1
                            )
                            
                            # Count by tag
                            tag = annotation.tag
                            stats["annotations_by_tag"][tag] = (
                                stats["annotations_by_tag"].get(tag, 0) + 1
                            )
                    
                except Exception as e:
                    logger.warning(f"Error processing annotation stats for {annotations_file}: {e}")
            
            stats["images_with_annotations"] = images_with_annotations
            stats["images_with_conflicts"] = images_with_conflicts
            
            # Calculate average
            if images_with_annotations > 0:
                stats["average_annotations_per_image"] = round(
                    stats["total_annotations"] / images_with_annotations, 2
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate annotation statistics: {e}")
            return {"error": str(e)}

    def save_llm_predictions(self, predictions: LLMPrediction):
        """Save LLM predictions with error handling"""
        try:
            predictions_path = self.predictions_dir / f"{predictions.image_id}.json"

            # Convert to dict and serialize datetime
            predictions_data = predictions.dict()

            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                return obj

            predictions_data = serialize_datetime(predictions_data)

            with open(predictions_path, "w") as f:
                json.dump(predictions_data, f, indent=2, ensure_ascii=False)

            # Update metadata to indicate predictions exist
            metadata = self.get_image_metadata(predictions.image_id)
            if metadata:
                metadata.has_ai_predictions = True
                self.save_image_metadata(metadata)

            logger.info(f"Saved LLM predictions for image {predictions.image_id}")

        except Exception as e:
            logger.error(f"Failed to save LLM predictions: {e}")
            raise FileStorageError(f"Cannot save LLM predictions: {e}")

    def get_storage_stats(self) -> Dict:
        """Get comprehensive storage statistics"""
        try:
            stats = {
                "total_images": 0,
                "total_annotations": 0,
                "total_predictions": 0,
                "total_storage_mb": 0,
                "formats": {},
                "avg_resolution": {"width": 0, "height": 0},
            }

            images = self.list_images()
            stats["total_images"] = len(images)

            total_width = total_height = 0
            total_bytes = 0

            for metadata in images:
                # Count formats
                format_name = metadata.format
                stats["formats"][format_name] = stats["formats"].get(format_name, 0) + 1

                # Sum dimensions for average
                total_width += metadata.width
                total_height += metadata.height
                total_bytes += metadata.file_size

                # Count annotations
                stats["total_annotations"] += metadata.annotation_count

                # Count predictions
                if metadata.has_ai_predictions:
                    stats["total_predictions"] += 1

            # Calculate averages
            if len(images) > 0:
                stats["avg_resolution"]["width"] = total_width // len(images)
                stats["avg_resolution"]["height"] = total_height // len(images)

            stats["total_storage_mb"] = round(total_bytes / (1024 * 1024), 2)

            return stats

        except Exception as e:
            logger.error(f"Error calculating storage stats: {e}")
            return {"error": str(e)}

    def save_temporary_file(self, file_content: bytes, filename: str, content_type: str) -> TemporaryFileInfo:
        """
        Save file to temporary location for validation processing
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: MIME content type
            
        Returns:
            TemporaryFileInfo: Information about the temporary file
            
        Raises:
            FileStorageError: If temporary save fails
        """
        try:
            # Generate unique temporary filename
            temp_id = str(uuid4())
            file_ext = Path(filename).suffix.lower() or '.tmp'
            temp_filename = f"{temp_id}{file_ext}"
            temp_path = self.temp_dir / temp_filename
            
            # Write file content to temporary location
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            # Create temporary file info
            temp_info = TemporaryFileInfo(
                temp_path=str(temp_path),
                original_filename=filename,
                file_size=len(file_content),
                content_type=content_type,
                expires_at=datetime.utcnow() + timedelta(hours=1)  # Expire in 1 hour
            )
            
            # Track for cleanup
            self._temp_files[temp_id] = temp_info
            
            logger.info(f"Saved temporary file: {temp_path} (ID: {temp_id})")
            return temp_info
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {e}")
            raise FileStorageError(f"Cannot save temporary file: {e}")
    
    def move_temp_to_permanent(self, temp_info: TemporaryFileInfo, target_image_id: str) -> str:
        """
        Move temporary file to permanent storage
        
        Args:
            temp_info: Information about the temporary file
            target_image_id: ID for the permanent image
            
        Returns:
            str: Path to the permanent file
            
        Raises:
            FileStorageError: If move operation fails
        """
        try:
            temp_path = Path(temp_info.temp_path)
            
            if not temp_path.exists():
                raise FileStorageError(f"Temporary file not found: {temp_path}")
            
            # Determine file extension
            file_ext = Path(temp_info.original_filename).suffix.lower()
            if not file_ext:
                # Try to determine from content
                try:
                    with Image.open(temp_path) as img:
                        format_to_ext = {
                            "JPEG": ".jpg",
                            "PNG": ".png", 
                            "GIF": ".gif",
                            "BMP": ".bmp",
                        }
                        file_ext = format_to_ext.get(img.format, ".jpg")
                except:
                    file_ext = ".jpg"  # default
            
            # Create permanent storage path
            storage_filename = f"{target_image_id}{file_ext}"
            permanent_path = self.images_dir / storage_filename
            
            # Move file from temp to permanent location
            shutil.move(str(temp_path), str(permanent_path))
            
            # Remove from temp tracking
            temp_id = temp_path.stem.split('_')[0] if '_' in temp_path.stem else temp_path.stem
            if temp_id in self._temp_files:
                del self._temp_files[temp_id]
            
            logger.info(f"Moved temporary file to permanent storage: {permanent_path}")
            return str(permanent_path)
            
        except Exception as e:
            logger.error(f"Failed to move temporary file to permanent storage: {e}")
            raise FileStorageError(f"Cannot move temporary file: {e}")
    
    def cleanup_temporary_file(self, temp_info: TemporaryFileInfo):
        """
        Clean up a specific temporary file
        
        Args:
            temp_info: Information about the temporary file to clean up
        """
        try:
            temp_path = Path(temp_info.temp_path)
            
            if temp_path.exists():
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            
            # Remove from tracking
            temp_id = temp_path.stem.split('_')[0] if '_' in temp_path.stem else temp_path.stem
            if temp_id in self._temp_files:
                del self._temp_files[temp_id]
                
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {temp_info.temp_path}: {e}")
    
    def cleanup_expired_temp_files(self):
        """Clean up expired temporary files"""
        current_time = datetime.utcnow()
        expired_files = []
        
        for temp_id, temp_info in self._temp_files.items():
            if temp_info.expires_at and current_time > temp_info.expires_at:
                expired_files.append(temp_id)
        
        for temp_id in expired_files:
            temp_info = self._temp_files[temp_id]
            self.cleanup_temporary_file(temp_info)
        
        if expired_files:
            logger.info(f"Cleaned up {len(expired_files)} expired temporary files")
    
    def save_validated_image(
        self, 
        temp_info: TemporaryFileInfo, 
        validation_result: "ValidationResult"
    ) -> ImageMetadata:
        """
        Save a validated image from temporary storage to permanent storage
        
        Args:
            temp_info: Temporary file information
            validation_result: LLM validation result
            
        Returns:
            ImageMetadata: Complete metadata with validation info
            
        Raises:
            FileStorageError: If saving fails
        """
        try:
            # Generate unique image ID
            image_id = str(uuid4())
            
            # Move temporary file to permanent storage
            permanent_path = self.move_temp_to_permanent(temp_info, image_id)
            
            # Extract image metadata using PIL
            try:
                with Image.open(permanent_path) as img:
                    width, height = img.size
                    format_name = img.format or "UNKNOWN"
            except Exception as e:
                # Clean up on PIL failure
                try:
                    os.unlink(permanent_path)
                except:
                    pass
                raise FileStorageError(f"Invalid image file - PIL processing failed: {e}")
            
            # Calculate checksum for duplicate detection
            with open(permanent_path, 'rb') as f:
                file_content = f.read()
            checksum = hashlib.md5(file_content).hexdigest()
            
            # Create validation info
            validation_info = ImageValidationInfo(
                checksum=checksum,
                original_filename=temp_info.original_filename,
                sanitized_filename=Path(permanent_path).name,
                file_size_bytes=temp_info.file_size,
            )
            
            # Create metadata with validation result
            metadata = ImageMetadata(
                id=image_id,
                filename=Path(permanent_path).name,
                file_path=permanent_path,
                file_size=temp_info.file_size,
                width=width,
                height=height,
                format=format_name,
                validation_info=validation_info,
                processing_status="completed",
                llm_validation_result=validation_result
            )
            
            # Save metadata
            self.save_image_metadata(metadata)
            
            # Update checksum cache
            self._checksum_cache[checksum] = image_id
            
            logger.info(
                f"Successfully saved validated image {image_id} "
                f"({width}x{height}, {temp_info.file_size} bytes, "
                f"validation: {validation_result.valid})"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to save validated image: {e}")
            # Attempt cleanup
            try:
                self.cleanup_temporary_file(temp_info)
            except:
                pass
            raise FileStorageError(f"Cannot save validated image: {e}")
    
    def get_temp_files_stats(self) -> Dict:
        """Get statistics about temporary files"""
        current_time = datetime.utcnow()
        
        stats = {
            "total_temp_files": len(self._temp_files),
            "expired_temp_files": 0,
            "total_temp_size_mb": 0
        }
        
        for temp_info in self._temp_files.values():
            if temp_info.expires_at and current_time > temp_info.expires_at:
                stats["expired_temp_files"] += 1
            
            stats["total_temp_size_mb"] += temp_info.file_size / (1024 * 1024)
        
        stats["total_temp_size_mb"] = round(stats["total_temp_size_mb"], 2)
        
        return stats
