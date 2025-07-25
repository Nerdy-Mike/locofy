from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from models.validation_models import ValidationResult


class UIElementTag(str, Enum):
    """Supported UI element types for labeling - now loaded from JSON config"""

    BUTTON = "button"
    INPUT = "input"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    LINK = "link"
    IMAGE = "image"
    TEXT = "text"


class AnnotationStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    APPROVED = "approved"
    REJECTED = "rejected"


class ConflictType(str, Enum):
    OVERLAP = "overlap"
    DUPLICATE = "duplicate"
    TAG_MISMATCH = "tag_mismatch"


class Dimensions(BaseModel):
    """Image dimensions for validation"""

    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")


class BoundingBox(BaseModel):
    """Bounding box coordinates for UI element annotation"""

    x: float = Field(..., ge=0, description="X coordinate of top-left corner")
    y: float = Field(..., ge=0, description="Y coordinate of top-left corner")
    width: float = Field(..., gt=0, description="Width of the bounding box")
    height: float = Field(..., gt=0, description="Height of the bounding box")

    @validator("width", "height")
    def validate_positive_dimensions(cls, v):
        if v <= 0:
            raise ValueError("Width and height must be positive")
        return v

    def get_coordinates(self) -> dict:
        """Get all four corner coordinates"""
        return {
            "top_left": (self.x, self.y),
            "top_right": (self.x + self.width, self.y),
            "bottom_left": (self.x, self.y + self.height),
            "bottom_right": (self.x + self.width, self.y + self.height),
        }

    def calculate_area(self) -> float:
        """Calculate the area of the bounding box"""
        return self.width * self.height

    def intersects_with(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects with another"""
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.height < other.y
            or other.y + other.height < self.y
        )

    def calculate_iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union (IoU) with another bounding box"""
        if not self.intersects_with(other):
            return 0.0

        # Calculate intersection
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x + self.width, other.x + other.width)
        y_bottom = min(self.y + self.height, other.y + other.height)

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.calculate_area() + other.calculate_area() - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


class Annotation(BaseModel):
    """UI element annotation with bounding box and tag"""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique annotation ID"
    )
    image_id: str = Field(..., description="ID of the associated image")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    tag: UIElementTag = Field(..., description="UI element type")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score (for AI predictions)"
    )
    annotator: Optional[str] = Field(None, description="Annotator identifier")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Ground Truth Extensions
    status: AnnotationStatus = AnnotationStatus.DRAFT
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    conflicts_with: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AnnotationCreate(BaseModel):
    """Model for creating new annotations"""

    image_id: str = Field(..., description="ID of the associated image")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    tag: UIElementTag = Field(..., description="UI element type")
    annotator: Optional[str] = Field(None, description="Annotator identifier")


class AnnotationUpdate(BaseModel):
    """Model for updating existing annotations"""

    bounding_box: Optional[BoundingBox] = Field(
        None, description="Updated bounding box coordinates"
    )
    tag: Optional[UIElementTag] = Field(None, description="Updated UI element type")
    annotator: Optional[str] = Field(None, description="Updated annotator identifier")


class ImageValidationInfo(BaseModel):
    """Validation information for uploaded images"""

    checksum: str = Field(..., description="MD5 checksum of image file")
    original_filename: str = Field(..., description="Original filename from upload")
    sanitized_filename: str = Field(
        ..., description="Sanitized filename used for storage"
    )
    validation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When validation was performed"
    )
    file_size_bytes: int = Field(..., gt=0, description="Actual file size in bytes")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ImageMetadata(BaseModel):
    """Enhanced metadata for uploaded images with comprehensive validation"""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique image ID")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Storage path")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    format: str = Field(..., description="Image format (JPEG, PNG, etc.)")
    upload_time: datetime = Field(
        default_factory=datetime.utcnow, description="Upload timestamp"
    )
    annotation_count: int = Field(default=0, ge=0, description="Number of annotations")
    has_ai_predictions: bool = Field(
        default=False, description="Whether AI predictions exist"
    )

    # Enhanced validation and security fields
    validation_info: Optional[ImageValidationInfo] = Field(
        None, description="Detailed validation information"
    )
    processing_status: str = Field(
        default="completed", description="Processing status of the image"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if processing failed"
    )

    # LLM validation result
    llm_validation_result: Optional[ValidationResult] = Field(
        None, description="Result of LLM-based UI validation"
    )

    @validator("width", "height")
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError("Image dimensions must be positive")
        if v > 16384:  # Reasonable maximum
            raise ValueError("Image dimensions too large (max 16384 pixels)")
        return v

    @validator("file_size")
    def validate_file_size(cls, v):
        max_size = 10 * 1024 * 1024  # 10MB
        if v > max_size:
            raise ValueError(
                f"File size too large (max {max_size / (1024*1024):.1f}MB)"
            )
        return v

    @validator("format")
    def validate_format(cls, v):
        allowed_formats = ["JPEG", "PNG", "GIF", "BMP", "WEBP"]
        if v.upper() not in allowed_formats:
            raise ValueError(f"Unsupported image format: {v}")
        return v.upper()

    @validator("processing_status")
    def validate_processing_status(cls, v):
        allowed_statuses = ["pending", "processing", "completed", "failed"]
        if v not in allowed_statuses:
            raise ValueError(f"Invalid processing status: {v}")
        return v

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)"""
        return self.width / self.height if self.height > 0 else 0

    @property
    def resolution_mp(self) -> float:
        """Calculate resolution in megapixels"""
        return (self.width * self.height) / 1_000_000

    @property
    def is_portrait(self) -> bool:
        """Check if image is in portrait orientation"""
        return self.height > self.width

    @property
    def is_landscape(self) -> bool:
        """Check if image is in landscape orientation"""
        return self.width > self.height

    def get_size_info(self) -> dict:
        """Get comprehensive size information"""
        return {
            "dimensions": f"{self.width}x{self.height}",
            "aspect_ratio": round(self.aspect_ratio, 2),
            "resolution_mp": round(self.resolution_mp, 2),
            "file_size_mb": round(self.file_size / (1024 * 1024), 2),
            "orientation": (
                "portrait"
                if self.is_portrait
                else "landscape" if self.is_landscape else "square"
            ),
        }

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class LLMPrediction(BaseModel):
    """LLM prediction result for UI element detection"""

    image_id: str = Field(..., description="ID of the associated image")
    predictions: List[Annotation] = Field(
        ..., description="List of predicted annotations"
    )
    llm_model: str = Field(..., description="Name of the LLM model used")
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Prediction timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        protected_namespaces = ()


class DraftAnnotation(BaseModel):
    temp_id: str  # Temporary ID (e.g., "temp_1")
    bounding_box: BoundingBox  # Coordinates from drawing
    tag: Optional[UIElementTag] = None  # User-assigned tag
    created_at: datetime = Field(default_factory=datetime.now)


class AnnotationSession(BaseModel):
    image_id: str
    draft_annotations: List[DraftAnnotation] = Field(default_factory=list)
    is_drawing: bool = False
    current_temp_box: Optional[BoundingBox] = None

    def add_draft(self, bbox: BoundingBox) -> str:
        """Add new draft annotation and return temp ID"""
        temp_id = f"temp_{len(self.draft_annotations) + 1}"
        draft = DraftAnnotation(temp_id=temp_id, bounding_box=bbox)
        self.draft_annotations.append(draft)
        return temp_id

    def ready_to_save(self) -> bool:
        """Check if all drafts have tags assigned"""
        return len(self.draft_annotations) > 0 and all(
            draft.tag is not None for draft in self.draft_annotations
        )


class AnnotationRequest(BaseModel):
    bounding_box: BoundingBox
    tag: UIElementTag
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class BatchAnnotationRequest(BaseModel):
    image_id: str
    created_by: str
    annotations: List[AnnotationRequest]

    @validator("annotations")
    def validate_annotations_not_empty(cls, v):
        if not v:
            raise ValueError("At least one annotation is required")
        if len(v) > 50:  # MAX_ANNOTATIONS_PER_BATCH
            raise ValueError("Too many annotations in batch (max 50)")
        return v


class ConflictInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    annotation_id: str
    image_id: str
    conflicts_with: List[str]
    created_by: str
    conflict_type: ConflictType
    severity: float = Field(ge=0.0, le=1.0, description="Conflict severity (0.0-1.0)")
    detected_at: datetime = Field(default_factory=datetime.now)
    iou_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ValidationError(BaseModel):
    field: str
    message: str
    value: Optional[str] = None


class ValidationWarning(BaseModel):
    field: str
    message: str
    value: Optional[str] = None


class ValidationResult(BaseModel):
    valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationWarning] = Field(default_factory=list)
    conflicts: List[ConflictInfo] = Field(default_factory=list)


class BatchAnnotationResponse(BaseModel):
    saved_count: int
    annotation_ids: List[str]
    conflicts: List[ConflictInfo] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time: float


class AnnotationConfig(BaseModel):
    min_box_width: int = 10
    min_box_height: int = 10
    overlap_threshold: float = 0.5
    max_annotations_per_batch: int = 50
    batch_save_timeout: int = 30
    max_conflict_checks: int = 1000


def rebuild_models():
    """Rebuild models to resolve forward references after all models are loaded"""
    try:
        # Import here to avoid circular imports during module loading
        from models.validation_models import ValidationResult

        # Rebuild the model to resolve forward references
        ImageMetadata.model_rebuild()

    except ImportError:
        # ValidationResult not available yet, skip rebuild
        pass
