#!/usr/bin/env python3
"""
Command-line Evaluation Tool for UI Component Annotations

Evaluates LLM predictions against ground truth annotations and provides
detailed metrics for each UI component tag.

Input:
- A folder of ground truth annotation files (JSON)
- A folder of LLM prediction files (JSON)

Output:
For each tag (button, input, radio, dropdown), calculate:
- Total number of ground truth boxes
- Number of correctly predicted boxes
- Precision
- Recall
- F1-score

Usage:
    python evaluate_annotations.py --ground-truth ./ground_truth --predictions ./predictions
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class BoundingBox:
    """Utility class for bounding box operations"""

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def calculate_iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union (IoU) with another bounding box"""
        # Calculate intersection coordinates
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        # Check if there's no intersection
        if x1 >= x2 or y1 >= y2:
            return 0.0

        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = self.width * self.height
        area2 = other.width * other.height
        union = area1 + area2 - intersection

        # Avoid division by zero
        if union == 0:
            return 0.0

        return intersection / union


class AnnotationEvaluator:
    """Evaluates annotation predictions against ground truth"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.ui_tags = ["button", "input", "radio", "dropdown"]

    def load_ground_truth_file(self, file_path: Path) -> List[Dict]:
        """Load ground truth annotations from JSON file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Handle both array format and wrapper format
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "annotations" in data:
                return data["annotations"]
            else:
                print(f"⚠️  Warning: Unexpected format in {file_path}")
                return []
        except Exception as e:
            print(f"❌ Error loading ground truth file {file_path}: {e}")
            return []

    def load_predictions_file(self, file_path: Path) -> List[Dict]:
        """Load predictions from JSON file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Handle wrapper format (with "predictions" key)
            if isinstance(data, dict) and "predictions" in data:
                return data["predictions"]
            elif isinstance(data, list):
                return data
            else:
                print(f"⚠️  Warning: Unexpected format in {file_path}")
                return []
        except Exception as e:
            print(f"❌ Error loading predictions file {file_path}: {e}")
            return []

    def find_matches(
        self, ground_truth: List[Dict], predictions: List[Dict]
    ) -> Tuple[Set[int], Set[int]]:
        """Find matches between ground truth and predictions using IoU threshold"""
        matched_gt_indices = set()
        matched_pred_indices = set()

        for gt_idx, gt_annotation in enumerate(ground_truth):
            if gt_idx in matched_gt_indices:
                continue

            gt_bbox = BoundingBox(
                gt_annotation["bounding_box"]["x"],
                gt_annotation["bounding_box"]["y"],
                gt_annotation["bounding_box"]["width"],
                gt_annotation["bounding_box"]["height"],
            )
            gt_tag = gt_annotation["tag"]

            best_match_idx = None
            best_iou = 0.0

            for pred_idx, prediction in enumerate(predictions):
                if pred_idx in matched_pred_indices:
                    continue

                pred_tag = prediction["tag"]

                # Tags must match for a valid match
                if gt_tag != pred_tag:
                    continue

                pred_bbox = BoundingBox(
                    prediction["bounding_box"]["x"],
                    prediction["bounding_box"]["y"],
                    prediction["bounding_box"]["width"],
                    prediction["bounding_box"]["height"],
                )

                iou = gt_bbox.calculate_iou(pred_bbox)

                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_idx = pred_idx

            # If we found a good match, mark both as matched
            if best_match_idx is not None:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_match_idx)

        return matched_gt_indices, matched_pred_indices

    def evaluate_single_pair(self, gt_file: Path, pred_file: Path) -> Dict[str, Dict]:
        """Evaluate a single ground truth/prediction file pair"""

        ground_truth = self.load_ground_truth_file(gt_file)
        predictions = self.load_predictions_file(pred_file)

        # Group annotations by tag
        gt_by_tag = defaultdict(list)
        pred_by_tag = defaultdict(list)

        for i, annotation in enumerate(ground_truth):
            tag = annotation.get("tag", "unknown")
            if tag in self.ui_tags:
                gt_by_tag[tag].append((i, annotation))

        for i, prediction in enumerate(predictions):
            tag = prediction.get("tag", "unknown")
            if tag in self.ui_tags:
                pred_by_tag[tag].append((i, prediction))

        # Find matches for the entire file
        matched_gt_indices, matched_pred_indices = self.find_matches(
            ground_truth, predictions
        )

        # Calculate metrics per tag
        tag_metrics = {}

        for tag in self.ui_tags:
            gt_annotations = [ann for _, ann in gt_by_tag[tag]]
            pred_annotations = [pred for _, pred in pred_by_tag[tag]]

            # Count matches for this specific tag
            gt_indices_for_tag = {
                i for i, ann in enumerate(ground_truth) if ann.get("tag") == tag
            }
            pred_indices_for_tag = {
                i for i, pred in enumerate(predictions) if pred.get("tag") == tag
            }

            matched_gt_for_tag = matched_gt_indices.intersection(gt_indices_for_tag)
            matched_pred_for_tag = matched_pred_indices.intersection(
                pred_indices_for_tag
            )

            # Calculate metrics
            true_positives = len(
                matched_gt_for_tag
            )  # Ground truth annotations that were correctly predicted
            false_positives = len(pred_indices_for_tag) - len(
                matched_pred_for_tag
            )  # Predictions with no corresponding ground truth
            false_negatives = len(gt_indices_for_tag) - len(
                matched_gt_for_tag
            )  # Ground truth annotations that were missed

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            tag_metrics[tag] = {
                "ground_truth_boxes": len(gt_annotations),
                "predicted_boxes": len(pred_annotations),
                "correctly_predicted_boxes": true_positives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }

        return tag_metrics

    def evaluate_datasets(self, ground_truth_dir: Path, predictions_dir: Path) -> Dict:
        """Evaluate entire datasets"""

        if not ground_truth_dir.exists():
            raise FileNotFoundError(
                f"Ground truth directory not found: {ground_truth_dir}"
            )

        if not predictions_dir.exists():
            raise FileNotFoundError(
                f"Predictions directory not found: {predictions_dir}"
            )

        gt_files = list(ground_truth_dir.glob("*.json"))
        pred_files = list(predictions_dir.glob("*.json"))

        print(f"📁 Found {len(gt_files)} ground truth files")
        print(f"📁 Found {len(pred_files)} prediction files")

        # Match files by name (excluding extension)
        gt_file_map = {f.stem: f for f in gt_files}
        pred_file_map = {f.stem: f for f in pred_files}

        common_files = set(gt_file_map.keys()).intersection(set(pred_file_map.keys()))

        if not common_files:
            raise ValueError(
                "No matching files found between ground truth and predictions directories"
            )

        print(f"🔗 Found {len(common_files)} matching file pairs")

        # Initialize aggregated metrics
        aggregated_metrics = {}
        for tag in self.ui_tags:
            aggregated_metrics[tag] = {
                "total_ground_truth_boxes": 0,
                "total_predicted_boxes": 0,
                "total_correctly_predicted_boxes": 0,
                "total_true_positives": 0,
                "total_false_positives": 0,
                "total_false_negatives": 0,
                "files_processed": 0,
            }

        # Process each file pair
        processed_count = 0
        for file_id in sorted(common_files):
            gt_file = gt_file_map[file_id]
            pred_file = pred_file_map[file_id]

            try:
                file_metrics = self.evaluate_single_pair(gt_file, pred_file)

                # Aggregate metrics
                for tag in self.ui_tags:
                    tag_data = file_metrics[tag]
                    agg_data = aggregated_metrics[tag]

                    agg_data["total_ground_truth_boxes"] += tag_data[
                        "ground_truth_boxes"
                    ]
                    agg_data["total_predicted_boxes"] += tag_data["predicted_boxes"]
                    agg_data["total_correctly_predicted_boxes"] += tag_data[
                        "correctly_predicted_boxes"
                    ]
                    agg_data["total_true_positives"] += tag_data["true_positives"]
                    agg_data["total_false_positives"] += tag_data["false_positives"]
                    agg_data["total_false_negatives"] += tag_data["false_negatives"]

                    if (
                        tag_data["ground_truth_boxes"] > 0
                        or tag_data["predicted_boxes"] > 0
                    ):
                        agg_data["files_processed"] += 1

                processed_count += 1

                if processed_count % 10 == 0:
                    print(
                        f"✅ Processed {processed_count}/{len(common_files)} file pairs"
                    )

            except Exception as e:
                print(f"❌ Error processing {file_id}: {e}")
                continue

        # Calculate final aggregated metrics
        final_metrics = {}
        for tag in self.ui_tags:
            agg_data = aggregated_metrics[tag]

            total_tp = agg_data["total_true_positives"]
            total_fp = agg_data["total_false_positives"]
            total_fn = agg_data["total_false_negatives"]

            precision = (
                total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            )
            recall = (
                total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            )
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            final_metrics[tag] = {
                "total_ground_truth_boxes": agg_data["total_ground_truth_boxes"],
                "total_predicted_boxes": agg_data["total_predicted_boxes"],
                "correctly_predicted_boxes": agg_data[
                    "total_correctly_predicted_boxes"
                ],
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "files_with_tag": agg_data["files_processed"],
            }

        return {
            "tag_metrics": final_metrics,
            "evaluation_params": {
                "iou_threshold": self.iou_threshold,
                "files_processed": processed_count,
                "total_files_found": len(common_files),
            },
        }


def print_evaluation_report(results: Dict):
    """Print a comprehensive evaluation report"""

    print("\n" + "=" * 80)
    print("📊 UI COMPONENT ANNOTATION EVALUATION REPORT")
    print("=" * 80)

    params = results["evaluation_params"]
    print(
        f"📁 Files processed: {params['files_processed']}/{params['total_files_found']}"
    )
    print(f"🎯 IoU threshold: {params['iou_threshold']}")
    print()

    # Overall summary
    tag_metrics = results["tag_metrics"]
    total_gt = sum(
        metrics["total_ground_truth_boxes"] for metrics in tag_metrics.values()
    )
    total_pred = sum(
        metrics["total_predicted_boxes"] for metrics in tag_metrics.values()
    )
    total_correct = sum(
        metrics["correctly_predicted_boxes"] for metrics in tag_metrics.values()
    )

    overall_precision = total_correct / total_pred if total_pred > 0 else 0.0
    overall_recall = total_correct / total_gt if total_gt > 0 else 0.0
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    print("📈 OVERALL METRICS:")
    print(f"   Total Ground Truth Boxes: {total_gt}")
    print(f"   Total Predicted Boxes: {total_pred}")
    print(f"   Correctly Predicted Boxes: {total_correct}")
    print(f"   Overall Precision: {overall_precision:.3f}")
    print(f"   Overall Recall: {overall_recall:.3f}")
    print(f"   Overall F1-Score: {overall_f1:.3f}")
    print()

    # Per-tag detailed metrics
    print("🏷️  METRICS BY TAG:")
    print("-" * 80)
    print(
        f"{'Tag':<12} {'GT Boxes':<10} {'Pred Boxes':<12} {'Correct':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}"
    )
    print("-" * 80)

    for tag in ["button", "input", "radio", "dropdown"]:
        metrics = tag_metrics[tag]
        print(
            f"{tag:<12} {metrics['total_ground_truth_boxes']:<10} "
            f"{metrics['total_predicted_boxes']:<12} {metrics['correctly_predicted_boxes']:<8} "
            f"{metrics['precision']:<10.3f} {metrics['recall']:<8.3f} {metrics['f1_score']:<8.3f}"
        )

    print("-" * 80)
    print()

    # Performance insights
    print("💡 INSIGHTS:")
    best_f1_tag = max(tag_metrics.keys(), key=lambda t: tag_metrics[t]["f1_score"])
    worst_f1_tag = min(tag_metrics.keys(), key=lambda t: tag_metrics[t]["f1_score"])

    print(
        f"   🎯 Best performing tag: {best_f1_tag} (F1: {tag_metrics[best_f1_tag]['f1_score']:.3f})"
    )
    print(
        f"   ⚠️  Worst performing tag: {worst_f1_tag} (F1: {tag_metrics[worst_f1_tag]['f1_score']:.3f})"
    )

    # Identify potential issues
    for tag, metrics in tag_metrics.items():
        if metrics["precision"] < 0.5:
            print(
                f"   ⚠️  {tag}: Low precision ({metrics['precision']:.3f}) - many false positives"
            )
        if metrics["recall"] < 0.5:
            print(
                f"   ⚠️  {tag}: Low recall ({metrics['recall']:.3f}) - many missed detections"
            )
        if metrics["total_ground_truth_boxes"] == 0:
            print(f"   ℹ️  {tag}: No ground truth annotations found")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM predictions against ground truth annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_annotations.py --ground-truth ./test_datasets/ground_truth --predictions ./test_datasets/predictions
    python evaluate_annotations.py --ground-truth ./gt --predictions ./pred --iou-threshold 0.3
    python evaluate_annotations.py --ground-truth ./gt --predictions ./pred --output-json ./results.json
        """,
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to directory containing ground truth annotation files",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to directory containing LLM prediction files",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a prediction correct (default: 0.5)",
    )
    parser.add_argument(
        "--output-json", type=str, help="Path to save detailed results as JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output final summary, suppress progress messages",
    )

    args = parser.parse_args()

    try:
        # Initialize evaluator
        evaluator = AnnotationEvaluator(iou_threshold=args.iou_threshold)

        # Run evaluation
        if not args.quiet:
            print("🚀 Starting annotation evaluation...")

        results = evaluator.evaluate_datasets(
            Path(args.ground_truth), Path(args.predictions)
        )

        # Print report
        print_evaluation_report(results)

        # Save JSON output if requested
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Detailed results saved to: {args.output_json}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
