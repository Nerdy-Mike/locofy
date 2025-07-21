"""
Validation utilities for the UI Component Labeling System.

This module provides comprehensive validation for all data types and user inputs
according to the specifications in DATAFLOW.md.
"""

import hashlib
import os
import string
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, Field

from models.annotation_models import BoundingBox, ImageMetadata


class ValidationError(BaseModel):
    """Validation error with details"""

    field: str
    message: str
    value: Optional[str] = None


class ValidationWarning(BaseModel):
    """Validation warning with details"""

    field: str
    message: str
    value: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of validation operation"""

    valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationWarning] = []

    def add_error(self, field: str, message: str, value: str = None):
        """Add a validation error"""
        self.errors.append(ValidationError(field=field, message=message, value=value))
        self.valid = False

    def add_warning(self, field: str, message: str, value: str = None):
        """Add a validation warning"""
        self.warnings.append(
            ValidationWarning(field=field, message=message, value=value)
        )


class ImageValidationResult(BaseModel):
    """Detailed result of image validation"""

    valid: bool
    format: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    size: Optional[int] = None
    checksum: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = []


class UploadValidator:
    """Validates file uploads according to DATAFLOW.md specifications"""

    # File type validation
    ALLOWED_MIME_TYPES = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/bmp",
    ]

    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

    # Size constraints
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_IMAGE_DIMENSION = 100  # pixels
    MAX_IMAGE_DIMENSION = 8192  # pixels

    # Content validation
    MIN_FILE_SIZE = 1024  # 1KB minimum

    @classmethod
    def validate_upload_file(
        cls, file_content: bytes, filename: str, content_type: str
    ) -> ValidationResult:
        """
        Comprehensive file upload validation

        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: MIME content type

        Returns:
            ValidationResult with detailed feedback
        """
        result = ValidationResult(valid=True)

        # Validate filename
        cls._validate_filename(filename, result)

        # Validate content type
        cls._validate_content_type(content_type, result)

        # Validate file size
        cls._validate_file_size(file_content, result)

        # Validate as image
        image_validation = cls._validate_image_content(file_content, result)

        return result

    @classmethod
    def _validate_filename(cls, filename: str, result: ValidationResult):
        """Validate filename safety and format"""
        if not filename:
            result.add_error("filename", "Filename is required")
            return

        if len(filename) > 255:
            result.add_error(
                "filename", "Filename too long (max 255 characters)", filename
            )
            return

        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            result.add_error(
                "filename", "Invalid filename: contains path separators", filename
            )
            return

        # Check for dangerous characters
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*", "\0"]
        if any(char in filename for char in dangerous_chars):
            result.add_error(
                "filename", "Invalid filename: contains dangerous characters", filename
            )
            return

        # Validate extension
        file_ext = Path(filename).suffix.lower()
        if not file_ext:
            result.add_warning("filename", "No file extension detected")
        elif file_ext not in cls.ALLOWED_EXTENSIONS:
            result.add_error(
                "filename", f"Unsupported file extension: {file_ext}", file_ext
            )

    @classmethod
    def _validate_content_type(cls, content_type: str, result: ValidationResult):
        """Validate MIME content type"""
        if not content_type:
            result.add_warning("content_type", "No content type provided")
            return

        if content_type not in cls.ALLOWED_MIME_TYPES:
            result.add_error(
                "content_type",
                f"Unsupported content type: {content_type}. Allowed: {', '.join(cls.ALLOWED_MIME_TYPES)}",
                content_type,
            )

    @classmethod
    def _validate_file_size(cls, file_content: bytes, result: ValidationResult):
        """Validate file size constraints"""
        size = len(file_content)

        if size < cls.MIN_FILE_SIZE:
            result.add_error(
                "file_size",
                f"File too small (min {cls.MIN_FILE_SIZE} bytes)",
                str(size),
            )

        if size > cls.MAX_FILE_SIZE:
            result.add_error(
                "file_size",
                f"File too large (max {cls.MAX_FILE_SIZE / (1024*1024):.1f}MB)",
                f"{size / (1024*1024):.1f}MB",
            )

    @classmethod
    def _validate_image_content(
        cls, file_content: bytes, result: ValidationResult
    ) -> ImageValidationResult:
        """Validate image content and extract metadata"""
        try:
            # Try to open as image
            image = Image.open(BytesIO(file_content))

            # Validate dimensions
            width, height = image.size

            if width < cls.MIN_IMAGE_DIMENSION or height < cls.MIN_IMAGE_DIMENSION:
                result.add_error(
                    "image_dimensions",
                    f"Image too small (min {cls.MIN_IMAGE_DIMENSION}x{cls.MIN_IMAGE_DIMENSION} pixels)",
                    f"{width}x{height}",
                )

            if width > cls.MAX_IMAGE_DIMENSION or height > cls.MAX_IMAGE_DIMENSION:
                result.add_error(
                    "image_dimensions",
                    f"Image too large (max {cls.MAX_IMAGE_DIMENSION}x{cls.MAX_IMAGE_DIMENSION} pixels)",
                    f"{width}x{height}",
                )

            # Check aspect ratio (warn if unusual)
            aspect_ratio = width / height
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                result.add_warning(
                    "aspect_ratio",
                    f"Unusual aspect ratio: {aspect_ratio:.2f}",
                    f"{width}x{height}",
                )

            # Validate format
            if not image.format:
                result.add_warning("image_format", "Could not determine image format")

            # Calculate checksum for duplicate detection
            checksum = hashlib.md5(file_content).hexdigest()

            return ImageValidationResult(
                valid=len(result.errors) == 0,
                format=image.format,
                dimensions=(width, height),
                size=len(file_content),
                checksum=checksum,
            )

        except Exception as e:
            result.add_error("image_content", f"Invalid image file: {str(e)}")
            return ImageValidationResult(
                valid=False, error=str(e), size=len(file_content)
            )

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename for safe storage

        Args:
            filename: Original filename

        Returns:
            Safe filename suitable for filesystem storage
        """
        if not filename:
            return "unnamed_file"

        # Get basename to prevent path traversal
        filename = os.path.basename(filename)

        # Keep only safe characters
        safe_chars = string.ascii_letters + string.digits + ".-_"
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)

        # Limit length
        if len(sanitized) > 200:
            # Preserve extension if possible
            parts = sanitized.split(".")
            if len(parts) > 1:
                ext = parts[-1]
                base = ".".join(parts[:-1])
                sanitized = base[: 200 - len(ext) - 1] + "." + ext
            else:
                sanitized = sanitized[:200]

        # Ensure it's not empty after sanitization
        if not sanitized or sanitized == ".":
            sanitized = "sanitized_file"

        return sanitized


class AnnotationValidator:
    """Validates annotation data and coordinates"""

    @classmethod
    def validate_bounding_box(
        cls, bbox: BoundingBox, image_metadata: ImageMetadata
    ) -> ValidationResult:
        """
        Validate bounding box coordinates against image dimensions

        Args:
            bbox: Bounding box to validate
            image_metadata: Image metadata with dimensions

        Returns:
            ValidationResult with coordinate validation details
        """
        result = ValidationResult(valid=True)

        # Check basic coordinate validity
        if bbox.x < 0:
            result.add_error(
                "x_coordinate", "X coordinate cannot be negative", str(bbox.x)
            )

        if bbox.y < 0:
            result.add_error(
                "y_coordinate", "Y coordinate cannot be negative", str(bbox.y)
            )

        if bbox.width <= 0:
            result.add_error("width", "Width must be positive", str(bbox.width))

        if bbox.height <= 0:
            result.add_error("height", "Height must be positive", str(bbox.height))

        # Check bounds against image dimensions
        if bbox.x + bbox.width > image_metadata.width:
            result.add_error(
                "bounds",
                f"Bounding box extends beyond image width ({image_metadata.width})",
                f"x={bbox.x}, width={bbox.width}",
            )

        if bbox.y + bbox.height > image_metadata.height:
            result.add_error(
                "bounds",
                f"Bounding box extends beyond image height ({image_metadata.height})",
                f"y={bbox.y}, height={bbox.height}",
            )

        # Check minimum size (warn if very small)
        min_size = 5  # pixels
        if bbox.width < min_size or bbox.height < min_size:
            result.add_warning(
                "size",
                f"Very small bounding box (less than {min_size}px)",
                f"{bbox.width}x{bbox.height}",
            )

        return result

    @classmethod
    def validate_annotation_data(cls, annotation_data: dict) -> ValidationResult:
        """Validate annotation request data"""
        result = ValidationResult(valid=True)

        # Check required fields
        required_fields = ["bounding_box", "tag", "annotator"]
        for field in required_fields:
            if field not in annotation_data:
                result.add_error(field, f"Required field '{field}' is missing")

        # Validate annotator name
        if "annotator" in annotation_data:
            annotator = annotation_data["annotator"]
            if not annotator or not annotator.strip():
                result.add_error("annotator", "Annotator name is required")
            elif len(annotator.strip()) > 100:
                result.add_error(
                    "annotator", "Annotator name too long (max 100 characters)"
                )

        return result


class InputSanitizer:
    """Sanitizes user inputs for security"""

    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 1000) -> str:
        """Sanitize text input by removing dangerous characters"""
        if not text:
            return ""

        # Strip whitespace
        text = text.strip()

        # Remove null bytes and control characters
        text = "".join(
            char for char in text if ord(char) >= 32 or char in ["\n", "\r", "\t"]
        )

        # Limit length
        if len(text) > max_length:
            text = text[:max_length]

        return text

    @staticmethod
    def sanitize_filename_input(filename: str) -> str:
        """Sanitize filename input from users"""
        return UploadValidator.sanitize_filename(filename)
