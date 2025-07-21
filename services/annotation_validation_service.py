"""
Annotation Validation Service

Implements comprehensive validation logic for annotation batches including:
- Individual annotation validation (coordinates, size, tags)
- Cross-annotation validation (overlaps within batch)
- Conflict detection with existing annotations
"""

import time
from typing import List, Optional

from models.annotation_models import (
    Annotation,
    AnnotationConfig,
    AnnotationRequest,
    BoundingBox,
    ConflictInfo,
    ConflictType,
    Dimensions,
    ImageMetadata,
    ValidationError,
    ValidationResult,
    ValidationWarning,
)
from utils.file_storage import FileStorageManager


class AnnotationValidationService:
    """Service for validating annotation batches and detecting conflicts"""

    def __init__(
        self, storage_manager: FileStorageManager, config: AnnotationConfig = None
    ):
        self.storage_manager = storage_manager
        self.config = config or AnnotationConfig()

    def validate_single_annotation(
        self, annotation: AnnotationRequest, image_metadata: ImageMetadata
    ) -> List[ValidationError]:
        """Validate individual annotation against image constraints"""
        errors = []
        bbox = annotation.bounding_box

        # Size validation
        if (
            bbox.width < self.config.min_box_width
            or bbox.height < self.config.min_box_height
        ):
            errors.append(
                ValidationError(
                    field="bounding_box",
                    message=f"Box too small (minimum {self.config.min_box_width}x{self.config.min_box_height})",
                    value=f"{bbox.width}x{bbox.height}",
                )
            )

        # Boundary validation
        if (
            bbox.x < 0
            or bbox.y < 0
            or bbox.x + bbox.width > image_metadata.width
            or bbox.y + bbox.height > image_metadata.height
        ):
            errors.append(
                ValidationError(
                    field="bounding_box",
                    message="Box extends outside image boundaries",
                    value=f"Box: ({bbox.x},{bbox.y}) {bbox.width}x{bbox.height}, Image: {image_metadata.width}x{image_metadata.height}",
                )
            )

        # Reasonable size check (warning, not error)
        image_area = image_metadata.width * image_metadata.height
        box_area = bbox.width * bbox.height
        if box_area > image_area * 0.8:
            # This will be added as a warning in the batch validation
            pass

        return errors

    def detect_batch_overlaps(
        self, annotations: List[AnnotationRequest]
    ) -> List[ConflictInfo]:
        """Detect overlaps within the same annotation batch"""
        conflicts = []

        for i, ann1 in enumerate(annotations):
            for j, ann2 in enumerate(annotations[i + 1 :], i + 1):
                iou = ann1.bounding_box.calculate_iou(ann2.bounding_box)
                if iou > self.config.overlap_threshold:
                    conflicts.append(
                        ConflictInfo(
                            annotation_id=f"temp_{i}",
                            image_id="",  # Will be set by caller
                            conflicts_with=[f"temp_{j}"],
                            created_by="",  # Will be set by caller
                            conflict_type=ConflictType.OVERLAP,
                            severity=iou,
                            iou_score=iou,
                        )
                    )

        return conflicts

    def detect_existing_conflicts(
        self, new_annotations: List[AnnotationRequest], image_id: str, created_by: str
    ) -> List[ConflictInfo]:
        """Detect conflicts with existing annotations"""
        conflicts = []

        # Load existing annotations
        existing_annotations = self.storage_manager.load_annotations(image_id)
        if not existing_annotations:
            return conflicts

        # Check each new annotation against existing ones
        for i, new_annotation in enumerate(new_annotations):
            for existing in existing_annotations:
                iou = new_annotation.bounding_box.calculate_iou(existing.bounding_box)
                if iou > self.config.overlap_threshold:
                    conflicts.append(
                        ConflictInfo(
                            annotation_id=f"temp_{i}",
                            image_id=image_id,
                            conflicts_with=[existing.id],
                            created_by=created_by,
                            conflict_type=ConflictType.OVERLAP,
                            severity=iou,
                            iou_score=iou,
                        )
                    )

        return conflicts

    def validate_annotation_batch(
        self, annotations: List[AnnotationRequest], image_id: str, created_by: str
    ) -> ValidationResult:
        """Validate entire annotation batch with comprehensive checks"""
        start_time = time.time()
        errors = []
        warnings = []
        conflicts = []

        # Load image metadata for validation
        try:
            image_metadata = self.storage_manager.get_image_metadata(image_id)
        except Exception as e:
            errors.append(
                ValidationError(
                    field="image_id",
                    message=f"Could not load image metadata: {str(e)}",
                    value=image_id,
                )
            )
            return ValidationResult(valid=False, errors=errors)

        # 1. Individual annotation validation
        for i, annotation in enumerate(annotations):
            annotation_errors = self.validate_single_annotation(
                annotation, image_metadata
            )
            for error in annotation_errors:
                error.field = f"annotation_{i}.{error.field}"
                errors.append(error)

            # Check for reasonable size (add warnings)
            bbox = annotation.bounding_box
            image_area = image_metadata.width * image_metadata.height
            box_area = bbox.width * bbox.height
            if box_area > image_area * 0.8:
                warnings.append(
                    ValidationWarning(
                        field=f"annotation_{i}.bounding_box",
                        message="Box covers most of the image - is this intentional?",
                        value=f"Box area: {box_area}, Image area: {image_area}",
                    )
                )

        # 2. Cross-annotation validation (within batch)
        if len(annotations) > 1:
            batch_conflicts = self.detect_batch_overlaps(annotations)
            for conflict in batch_conflicts:
                conflict.image_id = image_id
                conflict.created_by = created_by
            conflicts.extend(batch_conflicts)

        # 3. Conflict with existing annotations
        existing_conflicts = self.detect_existing_conflicts(
            annotations, image_id, created_by
        )
        conflicts.extend(existing_conflicts)

        # Performance check
        processing_time = time.time() - start_time
        if len(conflicts) > self.config.max_conflict_checks:
            warnings.append(
                ValidationWarning(
                    field="performance",
                    message=f"Too many conflicts detected ({len(conflicts)}), some may be omitted",
                    value=str(len(conflicts)),
                )
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            conflicts=conflicts,
        )

    def calculate_quality_impact(
        self, new_annotations: List[AnnotationRequest], image_id: str
    ) -> dict:
        """Calculate how new annotations will impact quality metrics"""
        existing_annotations = self.storage_manager.load_annotations(image_id)
        existing_count = len(existing_annotations) if existing_annotations else 0
        new_count = len(new_annotations)

        # Calculate new annotation density
        try:
            image_metadata = self.storage_manager.get_image_metadata(image_id)
            image_area = image_metadata.width * image_metadata.height

            total_annotated_area = sum(
                ann.bounding_box.width * ann.bounding_box.height
                for ann in new_annotations
            )
            coverage_percentage = (total_annotated_area / image_area) * 100

        except Exception:
            coverage_percentage = 0.0

        return {
            "existing_annotation_count": existing_count,
            "new_annotation_count": new_count,
            "total_annotation_count": existing_count + new_count,
            "coverage_percentage": coverage_percentage,
            "density_per_1000px": (
                (new_count / (image_area / 1000)) if image_area > 0 else 0
            ),
        }
