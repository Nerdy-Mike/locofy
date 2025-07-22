from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Status of image validation process"""

    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    ERROR = "error"


class ProcessingStatus(str, Enum):
    """Status of MCP processing"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DetectedElement(BaseModel):
    """UI element detected by MCP service"""

    id: str = Field(..., description="Unique element ID")
    tag: str = Field(..., description="UI element tag (button, input, etc.)")
    bounding_box: dict = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    reasoning: Optional[str] = Field(None, description="Detection reasoning")
    model_version: str = Field(..., description="Model version used")
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MCPContext(BaseModel):
    """Context for MCP prediction requests"""

    previous_predictions: Optional[List[DetectedElement]] = Field(
        None, description="Previous predictions for context"
    )
    image_metadata: Optional[dict] = Field(
        None, description="Image metadata for context"
    )
    user_feedback: Optional[dict] = Field(None, description="User feedback for context")


class PredictionResponse(BaseModel):
    """Response from MCP prediction service"""

    prediction_id: str = Field(..., description="Unique prediction ID")
    image_id: str = Field(..., description="Associated image ID")
    elements: List[DetectedElement] = Field(
        default_factory=list, description="Detected UI elements"
    )
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")
    confidence_threshold: float = Field(
        default=0.5, description="Confidence threshold used"
    )
    total_elements: int = Field(
        ..., ge=0, description="Total number of elements detected"
    )
    status: ProcessingStatus = Field(..., description="Processing status")
    context_used: bool = Field(default=False, description="Whether context was used")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ValidationResult(BaseModel):
    """Result of LLM-based image validation"""

    valid: bool = Field(..., description="Whether the image passed validation")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score from LLM"
    )
    reason: str = Field(..., description="Explanation of validation decision")
    processing_time: float = Field(
        ..., gt=0, description="Time taken for validation in seconds"
    )
    status: ValidationStatus = Field(..., description="Overall validation status")
    llm_model: str = Field(..., description="LLM model used for validation")
    validation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When validation was performed"
    )

    # Optional fields for detailed analysis
    detected_elements: Optional[list] = Field(
        None, description="List of UI elements detected by LLM"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if validation failed due to technical issues"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class UIValidationRequest(BaseModel):
    """Request model for UI image validation"""

    image_path: str = Field(..., description="Path to the image file to validate")
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for acceptance",
    )
    timeout_seconds: int = Field(
        default=10, gt=0, le=60, description="Maximum time to wait for LLM response"
    )
    include_element_detection: bool = Field(
        default=False, description="Whether to include detailed UI element detection"
    )


class TemporaryFileInfo(BaseModel):
    """Information about temporary files during upload process"""

    temp_path: str = Field(..., description="Path to temporary file")
    original_filename: str = Field(..., description="Original filename from upload")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When temporary file was created"
    )
    expires_at: Optional[datetime] = Field(
        None, description="When temporary file should be cleaned up"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ValidationConfig(BaseModel):
    """Configuration for LLM validation service"""

    enabled: bool = Field(default=True, description="Whether LLM validation is enabled")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Default confidence threshold"
    )
    timeout_seconds: int = Field(
        default=10, gt=0, le=60, description="Default timeout for LLM calls"
    )
    max_image_size_mb: float = Field(
        default=5.0, gt=0, description="Maximum image size to send to LLM (MB)"
    )
    fallback_on_error: bool = Field(
        default=True,
        description="Whether to allow upload if LLM validation fails due to errors",
    )
    cache_validation_results: bool = Field(
        default=True, description="Whether to cache validation results by image hash"
    )
    temp_file_cleanup_interval: int = Field(
        default=3600,
        gt=0,
        description="How often to clean up temporary files (seconds)",
    )


# Rebuild annotation models to resolve forward references
def _rebuild_annotation_models():
    """Rebuild annotation models after ValidationResult is defined"""
    try:
        from models.annotation_models import rebuild_models

        rebuild_models()
    except ImportError:
        pass


# Call rebuild to resolve forward references
_rebuild_annotation_models()
