"""
Coordinate Diagnostics Tool

Helps distinguish between coordinate detection issues vs rendering issues
by providing multiple validation methods and visual comparisons.
"""

import io
import json
import os

# Add parent directories to path for imports
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from frontend.components.image_viewer import ImageViewer
from frontend.utils.api_client import APIClient


class CoordinateDiagnostics:
    """Diagnostic tools for coordinate validation and rendering verification"""

    def __init__(self, image: Image.Image, image_id: str):
        self.image = image
        self.image_id = image_id
        self.width, self.height = image.size

    def validate_coordinates(self, annotations: List[Dict]) -> Dict[str, List]:
        """
        Validate coordinate values against image boundaries

        Returns:
            Dict with validation results categorized by issue type
        """
        issues = {
            "out_of_bounds": [],
            "negative_coordinates": [],
            "zero_dimensions": [],
            "overlapping": [],
            "suspicious_aspect_ratios": [],
            "valid": [],
        }

        for i, annotation in enumerate(annotations):
            bbox = annotation["bounding_box"]
            x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

            # Check for negative coordinates
            if x < 0 or y < 0:
                issues["negative_coordinates"].append(
                    {
                        "index": i,
                        "annotation": annotation,
                        "issue": f"Negative coordinates: x={x}, y={y}",
                    }
                )
                continue

            # Check for zero dimensions
            if width <= 0 or height <= 0:
                issues["zero_dimensions"].append(
                    {
                        "index": i,
                        "annotation": annotation,
                        "issue": f"Invalid dimensions: {width}×{height}",
                    }
                )
                continue

            # Check if coordinates are out of image bounds
            if x + width > self.width or y + height > self.height:
                issues["out_of_bounds"].append(
                    {
                        "index": i,
                        "annotation": annotation,
                        "issue": f"Box extends beyond image: ({x}, {y}) {width}×{height} > {self.width}×{self.height}",
                    }
                )
                continue

            # Check for suspicious aspect ratios (very thin or very wide)
            aspect_ratio = width / height if height > 0 else float("inf")
            if aspect_ratio > 20 or aspect_ratio < 0.05:
                issues["suspicious_aspect_ratios"].append(
                    {
                        "index": i,
                        "annotation": annotation,
                        "issue": f"Unusual aspect ratio: {aspect_ratio:.2f} (width/height = {width}/{height})",
                    }
                )

            # If we reach here, the annotation seems valid
            issues["valid"].append(
                {
                    "index": i,
                    "annotation": annotation,
                    "computed_area": width * height,
                    "aspect_ratio": aspect_ratio,
                    "center": (x + width / 2, y + height / 2),
                }
            )

        # Check for overlapping annotations
        valid_annotations = [item["annotation"] for item in issues["valid"]]
        for i, ann1 in enumerate(valid_annotations):
            for j, ann2 in enumerate(valid_annotations[i + 1 :], i + 1):
                if self._check_overlap(ann1["bounding_box"], ann2["bounding_box"]):
                    issues["overlapping"].append(
                        {
                            "annotation1": {"index": i, "annotation": ann1},
                            "annotation2": {"index": j, "annotation": ann2},
                            "overlap_area": self._calculate_overlap_area(
                                ann1["bounding_box"], ann2["bounding_box"]
                            ),
                        }
                    )

        return issues

    def _check_overlap(self, bbox1: Dict, bbox2: Dict) -> bool:
        """Check if two bounding boxes overlap"""
        x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
        x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]

        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def _calculate_overlap_area(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate the overlapping area between two bounding boxes"""
        x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
        x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]

        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        return overlap_x * overlap_y

    def test_rendering_accuracy(
        self, annotations: List[Dict], scale_factors: List[float] = None
    ) -> Dict:
        """
        Test rendering accuracy at different scale factors

        Returns:
            Dict with rendering test results
        """
        if scale_factors is None:
            scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0]

        results = {
            "scale_tests": [],
            "coordinate_precision": {},
            "rendering_consistency": {},
        }

        for scale in scale_factors:
            # Calculate display dimensions
            display_width = int(self.width * scale)
            display_height = int(self.height * scale)

            # Test coordinate scaling accuracy
            scale_issues = []
            for i, annotation in enumerate(annotations):
                bbox = annotation["bounding_box"]

                # Scale coordinates
                scaled_x = bbox["x"] * scale
                scaled_y = bbox["y"] * scale
                scaled_width = bbox["width"] * scale
                scaled_height = bbox["height"] * scale

                # Check for rounding errors
                if scaled_width < 1 or scaled_height < 1:
                    scale_issues.append(
                        {
                            "annotation_index": i,
                            "issue": "Box becomes too small when scaled",
                            "original_size": f"{bbox['width']}×{bbox['height']}",
                            "scaled_size": f"{scaled_width:.2f}×{scaled_height:.2f}",
                        }
                    )

                # Check for coordinate precision loss
                reverse_scaled_x = scaled_x / scale
                reverse_scaled_y = scaled_y / scale

                precision_loss_x = abs(reverse_scaled_x - bbox["x"])
                precision_loss_y = abs(reverse_scaled_y - bbox["y"])

                if precision_loss_x > 1 or precision_loss_y > 1:
                    scale_issues.append(
                        {
                            "annotation_index": i,
                            "issue": "Significant precision loss in coordinate scaling",
                            "original": f"({bbox['x']}, {bbox['y']})",
                            "precision_loss": f"({precision_loss_x:.2f}, {precision_loss_y:.2f})",
                        }
                    )

            results["scale_tests"].append(
                {
                    "scale_factor": scale,
                    "display_size": f"{display_width}×{display_height}",
                    "issues": scale_issues,
                    "total_issues": len(scale_issues),
                }
            )

        return results

    def create_diagnostic_overlay(
        self,
        annotations: List[Dict],
        show_grid: bool = True,
        show_centers: bool = True,
        show_dimensions: bool = True,
    ) -> go.Figure:
        """
        Create a detailed diagnostic overlay showing:
        - Grid lines for coordinate reference
        - Annotation centers
        - Dimension labels
        - Coordinate values
        """
        fig = go.Figure()

        # Add base image
        fig.add_layout_image(
            dict(
                source=self.image,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=self.width,
                sizey=self.height,
                sizing="stretch",
                opacity=0.7,  # Slightly transparent to see overlays better
                layer="below",
            )
        )

        # Add grid lines for coordinate reference
        if show_grid:
            grid_spacing = min(self.width, self.height) // 10

            # Vertical grid lines
            for x in range(0, self.width + 1, grid_spacing):
                fig.add_shape(
                    type="line",
                    x0=x,
                    y0=0,
                    x1=x,
                    y1=self.height,
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dot"),
                )

                # Add coordinate labels
                fig.add_annotation(
                    x=x,
                    y=self.height - 20,
                    text=str(x),
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                )

            # Horizontal grid lines
            for y in range(0, self.height + 1, grid_spacing):
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=y,
                    x1=self.width,
                    y1=y,
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dot"),
                )

                # Add coordinate labels
                fig.add_annotation(
                    x=20,
                    y=y,
                    text=str(y),
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                )

        # Add annotations with enhanced diagnostic info
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFD700", "#FF69B4", "#00CED1"]

        for i, annotation in enumerate(annotations):
            bbox = annotation["bounding_box"]
            x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            color = colors[i % len(colors)]

            # Add bounding box rectangle
            fig.add_shape(
                type="rect",
                x0=x,
                y0=y,
                x1=x + width,
                y1=y + height,
                line=dict(color=color, width=3),
                fillcolor=color,
                opacity=0.2,
            )

            # Add center point
            if show_centers:
                center_x, center_y = x + width / 2, y + height / 2
                fig.add_trace(
                    go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode="markers",
                        marker=dict(color=color, size=10, symbol="cross"),
                        name=f"Annotation {i+1} Center",
                        showlegend=False,
                    )
                )

            # Add coordinate and dimension labels
            if show_dimensions:
                # Top-left coordinate
                fig.add_annotation(
                    x=x,
                    y=y - 10,
                    text=f"({x}, {y})",
                    showarrow=True,
                    arrowcolor=color,
                    font=dict(size=10, color=color),
                    bgcolor="white",
                    bordercolor=color,
                    arrowhead=2,
                )

                # Dimensions label
                fig.add_annotation(
                    x=x + width / 2,
                    y=y + height / 2,
                    text=f"{width}×{height}",
                    showarrow=False,
                    font=dict(size=12, color=color),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=color,
                )

                # Tag label
                tag = annotation.get("tag", "unknown")
                fig.add_annotation(
                    x=x + width + 5,
                    y=y,
                    text=f"#{i+1}: {tag}",
                    showarrow=False,
                    font=dict(size=11, color=color),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=color,
                )

        # Configure layout
        fig.update_layout(
            width=1000,
            height=700,
            title="Coordinate Diagnostics Overlay",
            xaxis=dict(
                range=[0, self.width],
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=False,
                title="X Coordinate (pixels)",
            ),
            yaxis=dict(
                range=[self.height, 0],  # Inverted Y-axis for image coordinates
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=False,
                title="Y Coordinate (pixels)",
            ),
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60),
        )

        return fig

    def generate_coordinate_report(
        self, annotations: List[Dict], predictions: List[Dict] = None
    ) -> Dict:
        """Generate comprehensive coordinate analysis report"""

        report = {
            "image_info": {
                "dimensions": f"{self.width}×{self.height}",
                "total_pixels": self.width * self.height,
                "aspect_ratio": self.width / self.height,
            },
            "annotation_analysis": {},
            "prediction_analysis": {},
            "comparison": {},
        }

        # Analyze annotations
        if annotations:
            validation_results = self.validate_coordinates(annotations)
            rendering_tests = self.test_rendering_accuracy(annotations)

            report["annotation_analysis"] = {
                "total_count": len(annotations),
                "validation_results": validation_results,
                "rendering_tests": rendering_tests,
                "coordinate_statistics": self._calculate_coordinate_stats(annotations),
            }

        # Analyze predictions if provided
        if predictions:
            pred_validation = self.validate_coordinates(predictions)
            pred_rendering = self.test_rendering_accuracy(predictions)

            report["prediction_analysis"] = {
                "total_count": len(predictions),
                "validation_results": pred_validation,
                "rendering_tests": pred_rendering,
                "coordinate_statistics": self._calculate_coordinate_stats(predictions),
            }

            # Compare annotations vs predictions
            if annotations:
                report["comparison"] = self._compare_annotations_vs_predictions(
                    annotations, predictions
                )

        return report

    def _calculate_coordinate_stats(self, annotations: List[Dict]) -> Dict:
        """Calculate statistical summary of coordinates"""
        if not annotations:
            return {}

        x_coords = [ann["bounding_box"]["x"] for ann in annotations]
        y_coords = [ann["bounding_box"]["y"] for ann in annotations]
        widths = [ann["bounding_box"]["width"] for ann in annotations]
        heights = [ann["bounding_box"]["height"] for ann in annotations]
        areas = [w * h for w, h in zip(widths, heights)]

        return {
            "x_coordinates": {
                "min": min(x_coords),
                "max": max(x_coords),
                "mean": np.mean(x_coords),
                "std": np.std(x_coords),
            },
            "y_coordinates": {
                "min": min(y_coords),
                "max": max(y_coords),
                "mean": np.mean(y_coords),
                "std": np.std(y_coords),
            },
            "dimensions": {
                "width": {
                    "min": min(widths),
                    "max": max(widths),
                    "mean": np.mean(widths),
                },
                "height": {
                    "min": min(heights),
                    "max": max(heights),
                    "mean": np.mean(heights),
                },
                "area": {"min": min(areas), "max": max(areas), "mean": np.mean(areas)},
            },
        }

    def _compare_annotations_vs_predictions(
        self, annotations: List[Dict], predictions: List[Dict]
    ) -> Dict:
        """Compare coordinate patterns between annotations and predictions"""

        comparison = {
            "coordinate_distribution": {},
            "potential_matches": [],
            "significant_differences": [],
        }

        # Find potential matches (annotations and predictions that might refer to the same element)
        for i, ann in enumerate(annotations):
            ann_bbox = ann["bounding_box"]
            ann_center = (
                ann_bbox["x"] + ann_bbox["width"] / 2,
                ann_bbox["y"] + ann_bbox["height"] / 2,
            )

            closest_pred = None
            min_distance = float("inf")

            for j, pred in enumerate(predictions):
                pred_bbox = pred["bounding_box"]
                pred_center = (
                    pred_bbox["x"] + pred_bbox["width"] / 2,
                    pred_bbox["y"] + pred_bbox["height"] / 2,
                )

                # Calculate Euclidean distance between centers
                distance = np.sqrt(
                    (ann_center[0] - pred_center[0]) ** 2
                    + (ann_center[1] - pred_center[1]) ** 2
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_pred = {
                        "index": j,
                        "prediction": pred,
                        "distance": distance,
                    }

            if closest_pred and min_distance < 100:  # Threshold for "close" matches
                comparison["potential_matches"].append(
                    {
                        "annotation": {"index": i, "annotation": ann},
                        "closest_prediction": closest_pred,
                        "center_distance": min_distance,
                        "coordinate_difference": {
                            "x": abs(
                                ann_bbox["x"]
                                - closest_pred["prediction"]["bounding_box"]["x"]
                            ),
                            "y": abs(
                                ann_bbox["y"]
                                - closest_pred["prediction"]["bounding_box"]["y"]
                            ),
                            "width": abs(
                                ann_bbox["width"]
                                - closest_pred["prediction"]["bounding_box"]["width"]
                            ),
                            "height": abs(
                                ann_bbox["height"]
                                - closest_pred["prediction"]["bounding_box"]["height"]
                            ),
                        },
                    }
                )

        return comparison


def create_diagnostics_page():
    """Main diagnostics page interface"""
    st.title("🔍 Coordinate Diagnostics Tool")
    st.markdown(
        """
    This tool helps you distinguish between **coordinate detection issues** and **rendering problems** 
    by providing comprehensive validation and visual debugging capabilities.
    """
    )

    # Load available images
    try:
        api_client = APIClient()
        images = api_client.get_image_list()

        if not images:
            st.warning("No images found. Please upload some images first.")
            return

        # Image selection
        selected_image = st.selectbox(
            "Select an image to analyze:",
            options=images,
            format_func=lambda x: x.get("filename", x.get("id", "Unknown")),
        )

        if not selected_image:
            return

        image_id = selected_image["id"]

        # Load image and annotation data
        image_data = api_client.get_image(image_id)
        image = Image.open(io.BytesIO(image_data))

        annotations = api_client.get_annotations(image_id)
        predictions = api_client.get_predictions(image_id)

        # Create diagnostics instance
        diagnostics = CoordinateDiagnostics(image, image_id)

        # Display image info
        st.subheader("📊 Image Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dimensions", f"{image.width}×{image.height}")
        with col2:
            st.metric("Annotations", len(annotations))
        with col3:
            st.metric("Predictions", len(predictions))

        # Diagnostic options
        st.subheader("🛠️ Diagnostic Options")

        diagnostic_tabs = st.tabs(
            [
                "📐 Coordinate Validation",
                "🎯 Rendering Tests",
                "📈 Visual Analysis",
                "📋 Detailed Report",
            ]
        )

        with diagnostic_tabs[0]:
            st.markdown("### Coordinate Validation")

            # Validate annotations
            if annotations:
                st.markdown("#### Manual Annotations Validation")
                ann_validation = diagnostics.validate_coordinates(annotations)

                for issue_type, issues in ann_validation.items():
                    if issues:
                        if issue_type == "valid":
                            st.success(f"✅ {len(issues)} valid annotations found")
                        else:
                            st.error(
                                f"❌ {issue_type.replace('_', ' ').title()}: {len(issues)} issues"
                            )

                            with st.expander(f"View {issue_type} details"):
                                for issue in issues:
                                    st.write(
                                        f"- **Issue:** {issue.get('issue', 'Unknown')}"
                                    )
                                    if "annotation" in issue:
                                        bbox = issue["annotation"]["bounding_box"]
                                        st.write(
                                            f"  **Coordinates:** ({bbox['x']}, {bbox['y']}) {bbox['width']}×{bbox['height']}"
                                        )

            # Validate predictions
            if predictions:
                st.markdown("#### AI Predictions Validation")
                pred_validation = diagnostics.validate_coordinates(predictions)

                for issue_type, issues in pred_validation.items():
                    if issues:
                        if issue_type == "valid":
                            st.success(f"✅ {len(issues)} valid predictions found")
                        else:
                            st.warning(
                                f"⚠️ {issue_type.replace('_', ' ').title()}: {len(issues)} issues"
                            )

        with diagnostic_tabs[1]:
            st.markdown("### Rendering Accuracy Tests")

            test_data = annotations + predictions
            if test_data:
                rendering_results = diagnostics.test_rendering_accuracy(test_data)

                st.markdown("#### Scale Factor Tests")
                for test in rendering_results["scale_tests"]:
                    scale = test["scale_factor"]
                    issues = test["total_issues"]

                    if issues == 0:
                        st.success(
                            f"✅ Scale {scale}: No issues ({test['display_size']})"
                        )
                    else:
                        st.warning(
                            f"⚠️ Scale {scale}: {issues} issues ({test['display_size']})"
                        )

                        with st.expander(f"View scale {scale} issues"):
                            for issue in test["issues"]:
                                st.write(
                                    f"- **Annotation {issue['annotation_index']}:** {issue['issue']}"
                                )
                                st.write(
                                    f"  **Details:** {issue.get('original_size', '')} → {issue.get('scaled_size', '')}"
                                )

        with diagnostic_tabs[2]:
            st.markdown("### Visual Analysis")

            # Diagnostic overlay options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_grid = st.checkbox("Show coordinate grid", value=True)
            with col2:
                show_centers = st.checkbox("Show annotation centers", value=True)
            with col3:
                show_dimensions = st.checkbox("Show dimensions", value=True)

            # Create diagnostic overlay
            if annotations or predictions:
                combined_data = annotations + predictions
                fig = diagnostics.create_diagnostic_overlay(
                    combined_data,
                    show_grid=show_grid,
                    show_centers=show_centers,
                    show_dimensions=show_dimensions,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Provide coordinate checking tools
                st.markdown("#### Manual Coordinate Verification")
                st.markdown(
                    "Click on the plot above to get precise coordinates, or enter coordinates below to verify:"
                )

                col1, col2 = st.columns(2)
                with col1:
                    check_x = st.number_input(
                        "X coordinate to verify",
                        min_value=0,
                        max_value=image.width,
                        value=100,
                    )
                    check_y = st.number_input(
                        "Y coordinate to verify",
                        min_value=0,
                        max_value=image.height,
                        value=100,
                    )

                with col2:
                    # Show what should be at this coordinate
                    st.write(f"**Checking coordinate ({check_x}, {check_y})**")

                    # Check if this point falls within any annotation
                    point_annotations = []
                    for i, ann in enumerate(annotations + predictions):
                        bbox = ann["bounding_box"]
                        if (
                            bbox["x"] <= check_x <= bbox["x"] + bbox["width"]
                            and bbox["y"] <= check_y <= bbox["y"] + bbox["height"]
                        ):
                            point_annotations.append(
                                f"Annotation {i+1}: {ann.get('tag', 'unknown')}"
                            )

                    if point_annotations:
                        st.write("**Point is inside:**")
                        for ann in point_annotations:
                            st.write(f"- {ann}")
                    else:
                        st.write("Point is not inside any annotation")

        with diagnostic_tabs[3]:
            st.markdown("### Comprehensive Report")

            if annotations or predictions:
                report = diagnostics.generate_coordinate_report(
                    annotations, predictions
                )

                # Display report sections
                st.json(report)

                # Provide downloadable report
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="📄 Download Full Report",
                    data=report_json,
                    file_name=f"coordinate_diagnostics_{image_id}.json",
                    mime="application/json",
                )

        # Summary and recommendations
        st.subheader("💡 Diagnostic Summary & Recommendations")

        if annotations or predictions:
            total_issues = 0

            # Count total validation issues
            if annotations:
                ann_validation = diagnostics.validate_coordinates(annotations)
                for issue_type, issues in ann_validation.items():
                    if issue_type != "valid":
                        total_issues += len(issues)

            if predictions:
                pred_validation = diagnostics.validate_coordinates(predictions)
                for issue_type, issues in pred_validation.items():
                    if issue_type != "valid":
                        total_issues += len(issues)

            if total_issues == 0:
                st.success(
                    """
                ✅ **All coordinates appear valid!** 
                
                If you're experiencing rendering issues, they're likely related to:
                - Display scaling problems
                - Browser rendering inconsistencies
                - CSS/styling issues
                - Image format or compression artifacts
                """
                )
            else:
                st.warning(
                    f"""
                ⚠️ **Found {total_issues} coordinate issues**
                
                **Likely causes of coordinate problems:**
                - Incorrect coordinate system assumptions (origin position, Y-axis direction)
                - Scale factor miscalculations during annotation creation
                - Data format inconsistencies
                - Off-by-one errors in pixel calculations
                
                **Recommended next steps:**
                1. Check coordinate system documentation
                2. Verify scale factor calculations
                3. Test with known-good reference coordinates
                4. Validate annotation creation pipeline
                """
                )

    except Exception as e:
        st.error(f"❌ Error loading diagnostics: {str(e)}")
        st.code(f"Error details: {str(e)}")


if __name__ == "__main__":
    create_diagnostics_page()
