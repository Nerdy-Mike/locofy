"""
Quality Metrics Service

Handles calculation and management of annotation quality metrics including:
- Inter-annotator agreement calculation
- Quality score computation
- Conflict severity assessment
- Annotation density analysis
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from models.annotation_models import (
    Annotation,
    AnnotationStatus,
    ConflictInfo,
)
from utils.file_storage import FileStorageManager


class QualityMetricsService:
    """Service for calculating and managing annotation quality metrics"""

    def __init__(self, storage_manager: FileStorageManager):
        self.storage_manager = storage_manager

    def calculate_annotation_quality_score(
        self, annotations: List[Annotation]
    ) -> float:
        """
        Calculate quality score for a set of annotations

        Args:
            annotations: List of annotations to evaluate

        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if not annotations:
            return 0.0

        # Factors for quality calculation
        total_annotations = len(annotations)
        conflicted_count = sum(
            1 for ann in annotations if ann.status == AnnotationStatus.CONFLICTED
        )
        approved_count = sum(
            1 for ann in annotations if ann.status == AnnotationStatus.APPROVED
        )

        # Calculate base score from conflict ratio
        conflict_penalty = (
            conflicted_count / total_annotations if total_annotations > 0 else 0
        )
        base_score = 1.0 - conflict_penalty

        # Bonus for approved annotations
        approval_bonus = (
            (approved_count / total_annotations) * 0.1 if total_annotations > 0 else 0
        )

        # Density bonus (more annotations generally indicates better coverage)
        density_bonus = min(
            0.1, total_annotations / 100
        )  # Cap at 0.1 for 100+ annotations

        final_score = min(1.0, base_score + approval_bonus + density_bonus)
        return round(final_score, 3)

    def calculate_inter_annotator_agreement(
        self, annotations: List[Annotation]
    ) -> Optional[float]:
        """
        Calculate inter-annotator agreement for overlapping annotations

        Args:
            annotations: List of annotations from multiple annotators

        Returns:
            float: Agreement score between 0.0 and 1.0, or None if insufficient data
        """
        if len(annotations) < 2:
            return None

        # Group annotations by annotator
        annotator_groups = {}
        for ann in annotations:
            annotator = ann.created_by
            if annotator not in annotator_groups:
                annotator_groups[annotator] = []
            annotator_groups[annotator].append(ann)

        if len(annotator_groups) < 2:
            return None

        # Calculate pairwise agreements
        total_comparisons = 0
        total_agreement = 0.0

        annotators = list(annotator_groups.keys())
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                ann1_list = annotator_groups[annotators[i]]
                ann2_list = annotator_groups[annotators[j]]

                # Compare each annotation from annotator i with each from annotator j
                for ann1 in ann1_list:
                    for ann2 in ann2_list:
                        total_comparisons += 1

                        # Calculate IoU
                        iou = ann1.bounding_box.calculate_iou(ann2.bounding_box)

                        # Check tag agreement
                        tag_match = ann1.tag == ann2.tag

                        # Agreement score combines spatial overlap and tag agreement
                        if iou > 0.5 and tag_match:
                            total_agreement += 1.0
                        elif iou > 0.3 and tag_match:
                            total_agreement += 0.7
                        elif iou > 0.5:  # Good spatial overlap but wrong tag
                            total_agreement += 0.5
                        elif tag_match and iou > 0.1:  # Same tag but poor overlap
                            total_agreement += 0.3
                        # else: no agreement (0.0)

        if total_comparisons == 0:
            return None

        agreement_score = total_agreement / total_comparisons
        return round(agreement_score, 3)

    def update_image_quality_metrics(self, image_id: str) -> Optional[Dict]:
        """
        Update quality metrics for a specific image

        Args:
            image_id: ID of the image to update metrics for

        Returns:
            Dict: Updated quality metrics or None if error
        """
        try:
            # Load annotations for the image
            annotations = self.storage_manager.get_annotations(image_id)

            if not annotations:
                return None

            # Calculate quality metrics
            quality_score = self.calculate_annotation_quality_score(annotations)
            agreement_score = self.calculate_inter_annotator_agreement(annotations)

            # Count annotators
            unique_annotators = set(ann.created_by for ann in annotations)
            annotator_count = len(unique_annotators)

            # Count conflicts
            conflict_count = sum(
                1 for ann in annotations if ann.status == AnnotationStatus.CONFLICTED
            )
            has_conflicts = conflict_count > 0

            # Create quality metrics
            quality_metrics = {
                "image_id": image_id,
                "annotation_count": len(annotations),
                "annotator_count": annotator_count,
                "quality_score": quality_score,
                "agreement_score": agreement_score,
                "has_conflicts": has_conflicts,
                "conflict_count": conflict_count,
                "last_updated": datetime.utcnow().isoformat(),
            }

            # Save quality metrics to file
            self._save_quality_metrics(image_id, quality_metrics)

            # Update image metadata with quality info
            metadata = self.storage_manager.get_image_metadata(image_id)
            if metadata:
                metadata.quality_score = quality_score
                metadata.has_conflicts = has_conflicts
                metadata.annotation_count = len(annotations)
                self.storage_manager.save_image_metadata(metadata)

            return quality_metrics

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to update quality metrics for image {image_id}: {e}")
            return None

    def _save_quality_metrics(self, image_id: str, metrics: Dict):
        """Save quality metrics to file"""
        try:
            quality_dir = self.storage_manager.base_dir / "quality"
            quality_dir.mkdir(exist_ok=True)

            quality_path = quality_dir / f"{image_id}.json"

            import json

            with open(quality_path, "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save quality metrics for {image_id}: {e}")

    def get_quality_metrics(self, image_id: str) -> Optional[Dict]:
        """Load quality metrics for an image"""
        try:
            quality_path = (
                self.storage_manager.base_dir / "quality" / f"{image_id}.json"
            )

            if not quality_path.exists():
                return None

            import json

            with open(quality_path, "r") as f:
                return json.load(f)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load quality metrics for {image_id}: {e}")
            return None

    def get_system_quality_overview(self) -> Dict:
        """Get system-wide quality overview"""
        try:
            all_images = self.storage_manager.list_images()

            if not all_images:
                return {
                    "total_images": 0,
                    "total_annotations": 0,
                    "average_quality_score": 0.0,
                    "images_with_conflicts": 0,
                    "quality_distribution": {},
                }

            total_annotations = 0
            total_quality_score = 0.0
            images_with_quality = 0
            images_with_conflicts = 0
            quality_distribution = {"high": 0, "medium": 0, "low": 0, "unknown": 0}

            for image in all_images:
                total_annotations += image.annotation_count

                if image.has_conflicts:
                    images_with_conflicts += 1

                if image.quality_score is not None:
                    total_quality_score += image.quality_score
                    images_with_quality += 1

                    # Categorize quality
                    if image.quality_score >= 0.8:
                        quality_distribution["high"] += 1
                    elif image.quality_score >= 0.6:
                        quality_distribution["medium"] += 1
                    elif image.quality_score >= 0.3:
                        quality_distribution["low"] += 1
                    else:
                        quality_distribution["unknown"] += 1
                else:
                    quality_distribution["unknown"] += 1

            average_quality = (
                (total_quality_score / images_with_quality)
                if images_with_quality > 0
                else 0.0
            )

            return {
                "total_images": len(all_images),
                "total_annotations": total_annotations,
                "average_quality_score": round(average_quality, 3),
                "images_with_conflicts": images_with_conflicts,
                "quality_distribution": quality_distribution,
                "images_with_quality_scores": images_with_quality,
            }

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to calculate system quality overview: {e}")
            return {"error": str(e)}
