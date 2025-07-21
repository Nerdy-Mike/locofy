# Data Types Specification

## Overview

This document serves as the single source of truth for all data types used throughout the UI Component Labeling System. All components must adhere to these type definitions to ensure consistency and interoperability.

## 🔄 Recent Updates (Phase 1.1)

### Data Model Simplification
**Business Requirement**: Streamline the annotation process to focus only on core UI element classification.

**Changes Made:**
- ❌ **Removed `manual_name` fields** from all annotation models
- ✅ **Simplified workflow** to: Draw → Tag → Save
- ✅ **Four core UI element types**: button, input, radio, dropdown
- ✅ **Cleaner JSON output** without optional naming fields
- ✅ **Faster annotation process** with reduced cognitive load

**Affected Models:**
- `Annotation` - Removed `manual_name: Optional[str]` field
- `AnnotationRequest` - Removed `manual_name: Optional[str]` field  
- `DraftAnnotation` - Removed `manual_name: Optional[str]` field

This simplification aligns with business requirements to focus on essential UI element classification without complex manual naming workflows.

---

## Core Enums

### UIElementTag
```python
from enum import Enum

class UIElementTag(str, Enum):
    BUTTON = "button"
    INPUT = "input"
    RADIO = "radio"
    DROPDOWN = "dropdown"
```

### AnnotationStatus
```python
class AnnotationStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    APPROVED = "approved"
    REJECTED = "rejected"
```

### ConflictType
```python
class ConflictType(str, Enum):
    OVERLAP = "overlap"
    DUPLICATE = "duplicate"
    TAG_MISMATCH = "tag_mismatch"
```

### ProcessingStatus
```python
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### QualityLevel
```python
class QualityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"
```

---

## Basic Data Types

### BoundingBox
```python
from pydantic import BaseModel

class BoundingBox(BaseModel):
    x: float            # X coordinate (top-left corner)
    y: float            # Y coordinate (top-left corner)  
    width: float        # Width in pixels
    height: float       # Height in pixels
```

> **Why BoundingBox Format?**
> 
> This project uses the standard `(x, y, width, height)` format for consistency because:
> - **UI-Natural**: Matches how users draw annotations (click start point + drag)
> - **Industry Standard**: Used by COCO, YOLO, Pascal VOC, and most annotation tools
> - **Easy Validation**: Simple positive number checks (`width > 0`, `height > 0`)
> - **API Clarity**: Clear, intuitive field meanings in requests/responses
> - **ML Compatible**: Direct compatibility with most computer vision datasets

### Dimensions
```python
class Dimensions(BaseModel):
    width: int          # Width in pixels (integer for image dimensions)
    height: int         # Height in pixels (integer for image dimensions)
```

---

## 📐 Coordinate System & BoundingBox Math

### How BoundingBox Defines a Rectangle

A `BoundingBox` defines a rectangle using the **top-left corner** plus **dimensions**:

```python
# Example annotation
bbox = BoundingBox(x=100, y=50, width=200, height=150)

# This means:
# - Rectangle starts at pixel (100, 50) - top-left corner
# - Extends 200 pixels to the right (width)  
# - Extends 150 pixels downward (height)
# - Ends at pixel (300, 200) - bottom-right corner
```

### Visual Example
```
Image Grid (pixel coordinates):
     0    100         300
  0  ┌─────┬───────────┬────
     │     │           │
 50  ├─────●───────────●──── ← y=50 (top edge)
     │     │▓▓▓▓▓▓▓▓▓▓▓│     ▲
     │     │▓▓▓▓▓▓▓▓▓▓▓│     │ height=150
     │     │▓▓▓▓▓▓▓▓▓▓▓│     │
200  ├─────●───────────●──── ▼ y+height=200 (bottom edge)
     │     ▲           ▲
     │     x=100       x+width=300
     │     ◄─ width=200 ─►

BoundingBox(x=100, y=50, width=200, height=150)
```

### Corner Calculations (When Needed)

For geometric operations like IoU detection, calculate corners inline:

```python
def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    # Calculate corners from BoundingBox format
    left1, top1 = bbox1.x, bbox1.y
    right1, bottom1 = bbox1.x + bbox1.width, bbox1.y + bbox1.height
    
    left2, top2 = bbox2.x, bbox2.y  
    right2, bottom2 = bbox2.x + bbox2.width, bbox2.y + bbox2.height
    
    # Find intersection rectangle
    intersection_left = max(left1, left2)
    intersection_top = max(top1, top2)
    intersection_right = min(right1, right2)
    intersection_bottom = min(bottom1, bottom2)
    
    # Calculate IoU...
```

### Coordinate System Rules
- **Origin (0,0)**: Top-left corner of the image
- **X-axis**: Increases left → right
- **Y-axis**: Increases top → bottom (standard image coordinates)
- **All coordinates**: Relative to original image dimensions (not thumbnails)

---

## Image Types

### ImageMetadata
```python
from datetime import datetime
from typing import Optional

class ImageMetadata(BaseModel):
    id: str                                 # Unique identifier (UUID)
    filename: str                           # Original filename
    original_size: int                      # File size in bytes
    dimensions: Dimensions                  # Image dimensions
    format: str                            # Image format (JPEG, PNG, etc.)
    upload_timestamp: datetime             # Upload timestamp
    checksum: str                          # File hash for integrity
    processing_status: ProcessingStatus
    
    # Quality tracking
    annotation_count: int                  # Number of annotations
    quality_score: Optional[float] = None  # 0.0 to 1.0, optional
    has_conflicts: bool                    # Conflict detection flag
    thumbnail_path: Optional[str] = None   # Path to generated thumbnail
```

### ImageValidationResult
```python
class ImageValidationResult(BaseModel):
    valid: bool
    format: Optional[str] = None
    dimensions: Optional[Dimensions] = None
    size: Optional[int] = None
    error: Optional[str] = None
```

---

## Annotation Types

### Annotation (Simplified Model)
```python
from typing import List

class Annotation(BaseModel):
    id: str                                    # Unique identifier (UUID)
    image_id: str                             # Reference to image
    bounding_box: BoundingBox                 # Element coordinates
    tag: UIElementTag                         # UI element classification (button, input, radio, dropdown)
    confidence: Optional[float] = None        # 0.0 to 1.0, optional
    created_by: str                           # Annotator identifier
    created_at: datetime                      # Creation timestamp
    
    # Ground Truth Extensions
    status: AnnotationStatus                  # Current annotation status
    reviewed_by: Optional[str] = None         # Reviewer identifier
    reviewed_at: Optional[datetime] = None    # Review timestamp
    conflicts_with: List[str] = []           # IDs of conflicting annotations
    reasoning: Optional[str] = None           # Optional explanation
```

### AnnotationRequest (Simplified Model)
```python
class AnnotationRequest(BaseModel):
    bounding_box: BoundingBox                 # Element coordinates
    tag: UIElementTag                         # Required: button, input, radio, dropdown
    confidence: Optional[float] = None        # 0.0 to 1.0, optional
    reasoning: Optional[str] = None           # Optional explanation
```

### AnnotationResponse
```python
class AnnotationResponse(BaseModel):
    annotation_id: str
    status: AnnotationStatus
    conflicts: List[str] = []                # IDs of conflicting annotations
    quality_score: Optional[float] = None    # Calculated quality metric
    message: Optional[str] = None            # Status message
```

---

## Quality & Conflict Types

### QualityMetrics
```python
class QualityMetrics(BaseModel):
    image_id: str
    annotation_count: int                     # Total annotations for image
    annotator_count: int                      # Number of unique annotators
    agreement_score: Optional[float] = None   # Inter-annotator agreement (0.0-1.0)
    has_conflicts: bool                       # Conflict detection flag
    last_updated: datetime                    # Last update timestamp
    quality_level: QualityLevel               # Calculated quality assessment
```

### ConflictInfo
```python
class ConflictInfo(BaseModel):
    id: str                                   # Conflict identifier
    annotation_id: str                        # Primary annotation ID
    image_id: str                            # Reference to image
    conflicts_with: List[str]                 # IDs of conflicting annotations
    created_by: str                          # Original annotator
    conflict_type: ConflictType               # Type of conflict detected
    severity: float                          # Conflict severity (0.0-1.0)
    detected_at: datetime                     # Detection timestamp
    iou_score: Optional[float] = None         # Intersection over Union score
```

### ConflictResolution
```python
from typing import Literal

class ConflictResolution(BaseModel):
    conflict_id: str
    action: Literal["approve", "reject", "merge", "modify"]
    resolved_by: str                         # Resolver identifier
    resolved_at: datetime                     # Resolution timestamp
    reason: Optional[str] = None             # Resolution explanation
    resulting_annotation_id: Optional[str] = None  # Final annotation after resolution
```

### AgreementScore
```python
class AgreementScore(BaseModel):
    image_id: str
    score: float                             # Agreement score (0.0-1.0)
    annotator_count: int                     # Number of annotators involved
    annotation_pairs: int                    # Number of annotation pairs compared
    calculation_method: str                  # Algorithm used for calculation
    calculated_at: datetime                  # Calculation timestamp
```

---

## LLM & Prediction Types

### DetectedElement
```python
class DetectedElement(BaseModel):
    id: str                                  # Unique identifier
    tag: UIElementTag                        # Predicted UI element type
    bounding_box: BoundingBox                # Predicted coordinates
    confidence: float                        # Model confidence (0.0-1.0)
    reasoning: Optional[str] = None          # Model explanation
    model_version: str                       # LLM model identifier
    detection_timestamp: datetime            # Detection timestamp
    
    # Quality tracking
    verified: bool = False                   # Manual verification status
    verification_source: Optional[str] = None # Who verified the prediction
```

### PredictionRequest
```python
class PredictionRequest(BaseModel):
    image_id: str
    model_version: Optional[str] = None     # Optional model specification
    context: Optional[str] = None           # Additional context for prediction
```

### PredictionResponse
```python
class PredictionResponse(BaseModel):
    prediction_id: str
    image_id: str
    elements: List[DetectedElement]
    processing_time: float                  # Seconds
    model_version: str
    confidence_threshold: float
    total_elements: int
    status: ProcessingStatus
```

### MCPContext
```python
# Forward reference for UserFeedback
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .user_types import UserFeedback

class MCPContext(BaseModel):
    session_id: str
    image_id: str
    task_type: str
    user_feedback: Optional['UserFeedback'] = None
    previous_predictions: Optional[List[DetectedElement]] = None
```

---

## API Types

### ApiResponse
```python
from typing import TypeVar, Generic
T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime
```

### PaginatedResponse
```python
class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
```

### BatchOperation
```python
class BatchOperation(BaseModel):
    id: str                                 # Batch identifier
    operation_type: str                     # Type of batch operation
    total_items: int                        # Total items to process
    processed_items: int                    # Items completed
    failed_items: int                       # Items that failed
    status: ProcessingStatus                # Overall batch status
    created_at: datetime                    # Creation timestamp
    completed_at: Optional[datetime] = None # Completion timestamp
    error_log: List[str] = []              # Error messages
```

---

## Statistics & Analytics Types

### QualityDistribution
```python
class QualityDistribution(BaseModel):
    high: int
    medium: int
    low: int
    unknown: int
```

### QualityOverview
```python
class QualityOverview(BaseModel):
    total_images: int
    total_annotations: int
    average_agreement: float                 # 0.0 to 1.0
    conflict_count: int
    approval_rate: float                     # 0.0 to 1.0
    last_updated: datetime
    quality_distribution: QualityDistribution # Distribution by quality level
```

### AnnotatorStatistics
```python
class AnnotatorStatistics(BaseModel):
    annotator_id: str
    total_annotations: int
    approved_annotations: int
    conflicted_annotations: int
    average_confidence: float                # 0.0 to 1.0
    agreement_with_others: float             # 0.0 to 1.0
    annotations_per_day: float
    first_annotation: datetime
    last_annotation: datetime
```

### ModelPerformance
```python
class ModelPerformance(BaseModel):
    model_version: str
    total_predictions: int
    accuracy: float                          # 0.0 to 1.0
    precision: float                         # 0.0 to 1.0
    recall: float                           # 0.0 to 1.0
    f1_score: float                         # 0.0 to 1.0
    average_confidence: float                # 0.0 to 1.0
    processing_time_avg: float               # Average seconds
    evaluated_at: datetime
```

---

## Configuration Types

### SystemConfig
```python
class SystemConfig(BaseModel):
    overlap_threshold: float                 # IoU threshold for conflict detection
    min_agreement_score: float               # Minimum acceptable agreement
    auto_approve_threshold: float            # Auto-approve threshold
    require_review_votes: int                # Minimum votes for resolution
    max_file_size: int                      # Maximum upload size in bytes
    supported_formats: List[str]             # Allowed image formats
    thumbnail_size: int                      # Maximum thumbnail dimension
```

### UserFeedback
```python
class UserFeedback(BaseModel):
    type: Literal["correction", "validation", "suggestion"]
    annotation_id: str
    feedback_text: Optional[str] = None
    corrected_bbox: Optional[BoundingBox] = None
    corrected_tag: Optional[UIElementTag] = None
    provided_by: str
    provided_at: datetime
```

---

## File System Types

### FileReference
```python
class FileReference(BaseModel):
    path: str                               # Relative path from data root
    filename: str                           # Base filename
    size: int                              # File size in bytes
    created_at: datetime
    checksum: str                           # File integrity hash
```

### ExportIncludes
```python
class ExportIncludes(BaseModel):
    images: bool
    annotations: bool
    metadata: bool
    predictions: bool
    quality_metrics: bool
```

### ExportPackage
```python
class ExportPackage(BaseModel):
    id: str                                 # Export package identifier
    format: Literal["json", "zip", "csv"]
    includes: ExportIncludes
    file_references: List[FileReference]
    created_at: datetime
    expires_at: Optional[datetime] = None
```

---

## Validation Types

### ValidationError
```python
class ValidationError(BaseModel):
    field: str
    message: str
    value: Optional[str] = None
```

### ValidationWarning
```python
class ValidationWarning(BaseModel):
    field: str
    message: str
    value: Optional[str] = None
```

### ValidationRule
```python
from typing import Dict, Any

class ValidationRule(BaseModel):
    field: str
    rule_type: Literal["required", "range", "format", "custom"]
    parameters: Optional[Dict[str, Any]] = None
    error_message: str
```

### ValidationResult
```python
class ValidationResult(BaseModel):
    valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationWarning] = []
```

---

---

## Notes

- All timestamps use Python `datetime` objects 
- All numeric scores/percentages are represented as floats (0.0 to 1.0)
- UUIDs are used as strings for all primary identifiers
- Optional fields use `Optional[Type] = None`
- Lists use `List[Type]` annotation
- **BoundingBox coordinates**: Use `float` for sub-pixel precision in annotations
- **Dimensions**: Use `int` for image dimensions (always whole pixels)
- All coordinates are relative to original image dimensions (not thumbnails)
- File paths are relative to the data root directory (`/app/data/`)
- Use `Literal` for string enums with specific values
- Import required types: `from typing import Optional, List, Literal, Dict, Any`
- Import Pydantic: `from pydantic import BaseModel`
- Import datetime: `from datetime import datetime` 