"""
Quick Coordinate Validation Utility

A simple script to quickly validate coordinates and identify potential issues
with your annotation data. Run this to distinguish between coordinate detection
and rendering problems.
"""

import json
import os
from typing import Dict, List, Tuple

from PIL import Image


def load_annotation_data(
    image_id: str, data_dir: str = "data"
) -> Tuple[Dict, List, List, Dict]:
    """Load image metadata, annotations, predictions, and image"""

    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata", f"{image_id}.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load annotations
    annotations_path = os.path.join(data_dir, "annotations", f"{image_id}.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # Load predictions
    predictions_path = os.path.join(data_dir, "predictions", f"{image_id}.json")
    with open(predictions_path, "r") as f:
        predictions_data = json.load(f)
        predictions = predictions_data.get("predictions", [])

    # Load image
    image_path = os.path.join(data_dir, "images", f"{image_id}.png")
    image = Image.open(image_path)

    return metadata, annotations, predictions, image


def validate_coordinates(
    annotations: List[Dict],
    image_width: int,
    image_height: int,
    source_name: str = "annotations",
) -> Dict:
    """Validate coordinates against image boundaries and logical constraints"""

    print(f"\n🔍 Validating {len(annotations)} {source_name}...")

    issues = {
        "valid": [],
        "out_of_bounds": [],
        "negative_coordinates": [],
        "zero_dimensions": [],
        "suspicious_aspect_ratios": [],
        "very_small": [],
        "very_large": [],
    }

    for i, annotation in enumerate(annotations):
        bbox = annotation["bounding_box"]
        x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # Create annotation info for reporting
        ann_info = {
            "index": i,
            "tag": annotation.get("tag", "unknown"),
            "coordinates": f"({x}, {y})",
            "dimensions": f"{width}×{height}",
            "bbox": bbox,
        }

        has_issues = False

        # Check for negative coordinates
        if x < 0 or y < 0:
            issues["negative_coordinates"].append(
                {**ann_info, "issue": f"Negative coordinates: x={x}, y={y}"}
            )
            has_issues = True

        # Check for zero or negative dimensions
        if width <= 0 or height <= 0:
            issues["zero_dimensions"].append(
                {**ann_info, "issue": f"Invalid dimensions: {width}×{height}"}
            )
            has_issues = True

        # Check if coordinates are out of image bounds
        if x + width > image_width or y + height > image_height:
            issues["out_of_bounds"].append(
                {
                    **ann_info,
                    "issue": f"Extends beyond image: max_x={x+width} (limit: {image_width}), max_y={y+height} (limit: {image_height})",
                }
            )
            has_issues = True

        if not has_issues:
            # Check for suspicious characteristics (these don't invalidate the coordinates)
            aspect_ratio = width / height if height > 0 else float("inf")
            area = width * height

            # Very thin or very wide boxes
            if aspect_ratio > 50 or aspect_ratio < 0.02:
                issues["suspicious_aspect_ratios"].append(
                    {
                        **ann_info,
                        "issue": f"Unusual aspect ratio: {aspect_ratio:.2f}",
                        "aspect_ratio": aspect_ratio,
                    }
                )

            # Very small boxes (might disappear when scaled)
            elif area < 100:  # Less than 10x10 pixels
                issues["very_small"].append(
                    {
                        **ann_info,
                        "issue": f"Very small area: {area} pixels",
                        "area": area,
                    }
                )

            # Very large boxes (might indicate wrong detection)
            elif area > (image_width * image_height * 0.25):  # More than 25% of image
                issues["very_large"].append(
                    {
                        **ann_info,
                        "issue": f"Very large area: {area} pixels ({area/(image_width*image_height)*100:.1f}% of image)",
                        "area": area,
                    }
                )

            else:
                issues["valid"].append(ann_info)

    return issues


def test_scale_rendering(
    annotations: List[Dict], image_width: int, image_height: int
) -> Dict:
    """Test how annotations behave at different scale factors"""

    print(f"\n📐 Testing scale rendering...")

    scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = {}

    for scale in scale_factors:
        display_width = int(image_width * scale)
        display_height = int(image_height * scale)

        scale_issues = []

        for i, annotation in enumerate(annotations):
            bbox = annotation["bounding_box"]

            # Calculate scaled coordinates
            scaled_x = bbox["x"] * scale
            scaled_y = bbox["y"] * scale
            scaled_width = bbox["width"] * scale
            scaled_height = bbox["height"] * scale

            # Check for problems that occur during scaling
            if scaled_width < 1 or scaled_height < 1:
                scale_issues.append(
                    {
                        "annotation": i,
                        "tag": annotation.get("tag", "unknown"),
                        "issue": "Becomes invisible when scaled",
                        "original_size": f"{bbox['width']}×{bbox['height']}",
                        "scaled_size": f"{scaled_width:.2f}×{scaled_height:.2f}",
                    }
                )

            # Check for coordinate precision loss
            reverse_x = scaled_x / scale
            reverse_y = scaled_y / scale

            precision_loss_x = abs(reverse_x - bbox["x"])
            precision_loss_y = abs(reverse_y - bbox["y"])

            if precision_loss_x > 0.5 or precision_loss_y > 0.5:
                scale_issues.append(
                    {
                        "annotation": i,
                        "tag": annotation.get("tag", "unknown"),
                        "issue": "Coordinate precision loss",
                        "original": f"({bbox['x']}, {bbox['y']})",
                        "precision_loss": f"({precision_loss_x:.2f}, {precision_loss_y:.2f})",
                    }
                )

        results[scale] = {
            "display_size": f"{display_width}×{display_height}",
            "issues": scale_issues,
            "problematic_annotations": len(scale_issues),
        }

    return results


def compare_annotations_vs_predictions(
    annotations: List[Dict], predictions: List[Dict]
) -> Dict:
    """Compare manual annotations against AI predictions to identify systematic issues"""

    print(
        f"\n🆚 Comparing {len(annotations)} annotations vs {len(predictions)} predictions..."
    )

    if not annotations or not predictions:
        return {"message": "Need both annotations and predictions for comparison"}

    # Find potential matches (same UI element detected by both)
    matches = []
    unmatched_annotations = list(range(len(annotations)))
    unmatched_predictions = list(range(len(predictions)))

    for i, annotation in enumerate(annotations):
        ann_bbox = annotation["bounding_box"]
        ann_center = (
            ann_bbox["x"] + ann_bbox["width"] / 2,
            ann_bbox["y"] + ann_bbox["height"] / 2,
        )

        best_match = None
        min_distance = float("inf")

        for j, prediction in enumerate(predictions):
            pred_bbox = prediction["bounding_box"]
            pred_center = (
                pred_bbox["x"] + pred_bbox["width"] / 2,
                pred_bbox["y"] + pred_bbox["height"] / 2,
            )

            # Calculate distance between centers
            distance = (
                (ann_center[0] - pred_center[0]) ** 2
                + (ann_center[1] - pred_center[1]) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance = distance
                best_match = j

        # If we found a reasonably close match (within 200 pixels)
        if min_distance < 200:
            matches.append(
                {
                    "annotation_index": i,
                    "prediction_index": best_match,
                    "center_distance": min_distance,
                    "annotation": annotation,
                    "prediction": predictions[best_match],
                    "coordinate_differences": {
                        "x": abs(
                            ann_bbox["x"] - predictions[best_match]["bounding_box"]["x"]
                        ),
                        "y": abs(
                            ann_bbox["y"] - predictions[best_match]["bounding_box"]["y"]
                        ),
                        "width": abs(
                            ann_bbox["width"]
                            - predictions[best_match]["bounding_box"]["width"]
                        ),
                        "height": abs(
                            ann_bbox["height"]
                            - predictions[best_match]["bounding_box"]["height"]
                        ),
                    },
                }
            )

            if i in unmatched_annotations:
                unmatched_annotations.remove(i)
            if best_match in unmatched_predictions:
                unmatched_predictions.remove(best_match)

    return {
        "matches": matches,
        "unmatched_annotations": unmatched_annotations,
        "unmatched_predictions": unmatched_predictions,
        "match_rate": len(matches) / max(len(annotations), len(predictions)) * 100,
    }


def print_validation_results(issues: Dict, source_name: str):
    """Print validation results in a readable format"""

    total_annotations = sum(len(issue_list) for issue_list in issues.values())

    print(f"\n📊 {source_name.upper()} VALIDATION RESULTS:")
    print(f"{'='*50}")

    # Print summary
    valid_count = len(issues["valid"])
    invalid_count = total_annotations - valid_count

    if invalid_count == 0:
        print(f"✅ ALL {valid_count} {source_name} ARE VALID!")
        print(
            "   If you're seeing rendering issues, the problem is likely in the display/rendering code, not the coordinates."
        )
    else:
        print(
            f"❌ FOUND {invalid_count} ISSUES OUT OF {total_annotations} {source_name}"
        )
        print(f"✅ {valid_count} {source_name} are valid")

    # Print details of each issue type
    for issue_type, issue_list in issues.items():
        if issue_type == "valid" or not issue_list:
            continue

        print(f"\n🔴 {issue_type.replace('_', ' ').upper()}: {len(issue_list)} issues")

        for issue in issue_list[:5]:  # Show first 5 issues of each type
            print(
                f"   - {issue_type} #{issue['index']}: {issue.get('tag', 'unknown')} at {issue['coordinates']} size {issue['dimensions']}"
            )
            print(f"     Issue: {issue['issue']}")

        if len(issue_list) > 5:
            print(f"   ... and {len(issue_list) - 5} more {issue_type} issues")


def print_scale_results(scale_results: Dict):
    """Print scale testing results"""

    print(f"\n📐 SCALE RENDERING TEST RESULTS:")
    print(f"{'='*50}")

    for scale, results in scale_results.items():
        issue_count = results["problematic_annotations"]
        if issue_count == 0:
            print(f"✅ Scale {scale:4.1f}: No issues ({results['display_size']})")
        else:
            print(
                f"⚠️  Scale {scale:4.1f}: {issue_count} issues ({results['display_size']})"
            )

            for issue in results["issues"][:3]:  # Show first 3 issues
                print(
                    f"   - Annotation #{issue['annotation']} ({issue['tag']}): {issue['issue']}"
                )

            if len(results["issues"]) > 3:
                print(
                    f"   ... and {len(results['issues']) - 3} more issues at this scale"
                )


def print_comparison_results(comparison: Dict):
    """Print annotation vs prediction comparison results"""

    if "message" in comparison:
        print(f"\n🆚 COMPARISON: {comparison['message']}")
        return

    print(f"\n🆚 ANNOTATION vs PREDICTION COMPARISON:")
    print(f"{'='*50}")

    matches = comparison["matches"]
    match_rate = comparison["match_rate"]

    print(f"📊 Match Rate: {match_rate:.1f}%")
    print(f"🔗 Found {len(matches)} potential matches")
    print(
        f"❓ {len(comparison['unmatched_annotations'])} annotations have no close predictions"
    )
    print(
        f"❓ {len(comparison['unmatched_predictions'])} predictions have no close annotations"
    )

    if matches:
        print(f"\n📏 COORDINATE DIFFERENCES IN MATCHES:")

        # Calculate average differences
        avg_x_diff = sum(m["coordinate_differences"]["x"] for m in matches) / len(
            matches
        )
        avg_y_diff = sum(m["coordinate_differences"]["y"] for m in matches) / len(
            matches
        )
        avg_w_diff = sum(m["coordinate_differences"]["width"] for m in matches) / len(
            matches
        )
        avg_h_diff = sum(m["coordinate_differences"]["height"] for m in matches) / len(
            matches
        )

        print(f"   Average X difference: {avg_x_diff:.1f} pixels")
        print(f"   Average Y difference: {avg_y_diff:.1f} pixels")
        print(f"   Average Width difference: {avg_w_diff:.1f} pixels")
        print(f"   Average Height difference: {avg_h_diff:.1f} pixels")

        # Show biggest discrepancies
        print(f"\n🎯 LARGEST DISCREPANCIES:")
        sorted_matches = sorted(
            matches, key=lambda x: x["center_distance"], reverse=True
        )

        for match in sorted_matches[:3]:
            ann = match["annotation"]
            pred = match["prediction"]
            diffs = match["coordinate_differences"]

            print(
                f"   - Annotation #{match['annotation_index']} vs Prediction #{match['prediction_index']}"
            )
            print(
                f"     Tags: {ann.get('tag', 'unknown')} vs {pred.get('tag', 'unknown')}"
            )
            print(f"     Position diff: ({diffs['x']}, {diffs['y']}) pixels")
            print(f"     Size diff: {diffs['width']}×{diffs['height']} pixels")
            print(f"     Center distance: {match['center_distance']:.1f} pixels")


def main():
    """Run comprehensive coordinate validation on your current data"""

    print("🔍 COORDINATE VALIDATION TOOL")
    print("=" * 50)
    print("This tool will help you identify whether issues are with:")
    print("1. 🎯 Coordinate Detection (wrong coordinates)")
    print("2. 🖼️  Rendering Problems (correct coordinates, display issues)")
    print()

    # Use the image ID from your current data
    image_id = "15d02cb6-8498-4ad3-8169-058496bfab22"

    try:
        # Load all data
        print(f"📁 Loading data for image: {image_id}")
        metadata, annotations, predictions, image = load_annotation_data(image_id)

        print(f"📊 Loaded:")
        print(f"   - Image: {image.width}×{image.height} pixels")
        print(f"   - Annotations: {len(annotations)}")
        print(f"   - Predictions: {len(predictions)}")

        # Validate annotations
        if annotations:
            ann_issues = validate_coordinates(
                annotations, image.width, image.height, "annotations"
            )
            print_validation_results(ann_issues, "annotations")

        # Validate predictions
        if predictions:
            pred_issues = validate_coordinates(
                predictions, image.width, image.height, "predictions"
            )
            print_validation_results(pred_issues, "predictions")

        # Test scale rendering
        all_annotations = annotations + predictions
        if all_annotations:
            scale_results = test_scale_rendering(
                all_annotations, image.width, image.height
            )
            print_scale_results(scale_results)

        # Compare annotations vs predictions
        if annotations and predictions:
            comparison = compare_annotations_vs_predictions(annotations, predictions)
            print_comparison_results(comparison)

        # Final diagnosis and recommendations
        print(f"\n💡 DIAGNOSIS & RECOMMENDATIONS:")
        print(f"{'='*50}")

        total_coord_issues = 0
        if annotations:
            total_coord_issues += len(annotations) - len(ann_issues["valid"])
        if predictions:
            total_coord_issues += len(predictions) - len(pred_issues["valid"])

        if total_coord_issues == 0:
            print("✅ COORDINATE DETECTION: All coordinates appear valid!")
            print("   📋 Your coordinate detection is working correctly.")
            print("   🎯 If you see visual issues, focus on:")
            print("      - CSS styling and positioning")
            print("      - Scale factor calculations in rendering")
            print("      - Browser compatibility issues")
            print("      - Image format or compression problems")
        else:
            print(
                f"❌ COORDINATE DETECTION: Found {total_coord_issues} coordinate issues!"
            )
            print("   🔧 Priority fixes needed:")
            print("      - Review coordinate system assumptions")
            print("      - Check for off-by-one errors")
            print("      - Validate data input/conversion pipeline")
            print("      - Test coordinate calculations manually")

        # Scale rendering assessment
        problematic_scales = sum(
            1
            for results in scale_results.values()
            if results["problematic_annotations"] > 0
        )
        if problematic_scales > 0:
            print(f"\n⚠️  SCALE RENDERING: Issues at {problematic_scales} scale levels")
            print("   📐 Consider implementing:")
            print("      - Minimum size thresholds for small annotations")
            print("      - Better coordinate precision handling")
            print("      - Adaptive rendering based on scale")
        else:
            print(f"\n✅ SCALE RENDERING: All scales render correctly")

        print(
            f"\n✨ Analysis complete! Check the detailed output above for specific issues."
        )

    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        print(
            "   Please check that your data files exist and are in the expected format."
        )


if __name__ == "__main__":
    main()
