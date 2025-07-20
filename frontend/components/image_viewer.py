from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from PIL import Image

import streamlit as st


class ImageViewer:
    """Reusable component for displaying images with annotation overlays"""

    def __init__(self, image: Image.Image):
        self.image = image
        self.colors = {
            "button": "#FF0000",  # Red
            "input": "#0000FF",  # Blue
            "radio": "#00FF00",  # Green
            "dropdown": "#FFA500",  # Orange
        }

    def render_with_annotations(
        self,
        annotations: List[Dict] = None,
        predictions: List[Dict] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        width: int = 800,
        height: int = 600,
    ) -> go.Figure:
        """Render image with annotations and predictions overlaid"""

        fig = go.Figure()

        # Add the base image
        fig.add_layout_image(
            dict(
                source=self.image,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=self.image.width,
                sizey=self.image.height,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        )

        # Add ground truth annotations (solid lines)
        if annotations:
            for i, annotation in enumerate(annotations):
                self._add_annotation_to_figure(
                    fig, annotation, "Ground Truth", solid=True, show_labels=show_labels
                )

        # Add AI predictions (dashed lines)
        if predictions:
            for i, prediction in enumerate(predictions):
                self._add_annotation_to_figure(
                    fig,
                    prediction,
                    "AI Prediction",
                    solid=False,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                )

        # Configure layout
        fig.update_layout(
            width=width,
            height=height,
            xaxis=dict(
                range=[0, self.image.width],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                range=[self.image.height, 0],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode=False,
        )

        return fig

    def _add_annotation_to_figure(
        self,
        fig: go.Figure,
        annotation: Dict,
        annotation_type: str,
        solid: bool = True,
        show_labels: bool = True,
        show_confidence: bool = True,
    ):
        """Add a single annotation to the figure"""

        bbox = annotation["bounding_box"]
        tag = annotation["tag"]
        confidence = annotation.get("confidence")

        # Get color for this tag type
        color = self.colors.get(tag, "#000000")

        # Determine line style
        line_style = dict(color=color, width=2)
        if not solid:
            line_style["dash"] = "dash"

        # Add rectangle
        fig.add_shape(
            type="rect",
            x0=bbox["x"],
            y0=bbox["y"],
            x1=bbox["x"] + bbox["width"],
            y1=bbox["y"] + bbox["height"],
            line=line_style,
            fillcolor=color,
            opacity=0.1 if solid else 0.05,
        )

        # Add label if requested
        if show_labels:
            label_text = f"{annotation_type}: {tag}"
            if show_confidence and confidence is not None:
                label_text += f" ({confidence:.2f})"

            # Position label above or below rectangle depending on position
            label_y = (
                bbox["y"] - 15 if bbox["y"] > 20 else bbox["y"] + bbox["height"] + 15
            )

            fig.add_annotation(
                x=bbox["x"],
                y=label_y,
                text=label_text,
                showarrow=False,
                font=dict(color=color, size=10),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
                opacity=0.9,
            )

    def render_comparison_view(
        self,
        annotations: List[Dict],
        predictions: List[Dict],
        title: str = "Ground Truth vs AI Predictions",
    ):
        """Render side-by-side comparison of annotations and predictions"""

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ground Truth")
            fig_gt = self.render_with_annotations(
                annotations=annotations, show_labels=True, width=400, height=300
            )
            st.plotly_chart(fig_gt, use_container_width=True)

            # Statistics
            if annotations:
                summary = self._get_annotation_summary(annotations)
                st.write("**Component Counts:**")
                for tag, count in summary.items():
                    if count > 0:
                        st.write(f"- {tag.title()}: {count}")

        with col2:
            st.subheader("AI Predictions")
            fig_pred = self.render_with_annotations(
                predictions=predictions,
                show_labels=True,
                show_confidence=True,
                width=400,
                height=300,
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Statistics
            if predictions:
                summary = self._get_annotation_summary(predictions)
                avg_confidence = sum(p.get("confidence", 0) for p in predictions) / len(
                    predictions
                )

                st.write("**Component Counts:**")
                for tag, count in summary.items():
                    if count > 0:
                        st.write(f"- {tag.title()}: {count}")
                st.write(f"**Avg Confidence:** {avg_confidence:.2f}")

        # Combined view
        st.subheader("Combined View")
        fig_combined = self.render_with_annotations(
            annotations=annotations,
            predictions=predictions,
            show_labels=True,
            show_confidence=True,
            width=800,
            height=500,
        )

        # Add legend manually
        st.markdown(
            """
        **Legend:**
        - 🔴 **Red solid lines**: Ground truth annotations
        - 🔵 **Blue dashed lines**: AI predictions
        """
        )

        st.plotly_chart(fig_combined, use_container_width=True)

    def _get_annotation_summary(self, annotations: List[Dict]) -> Dict[str, int]:
        """Get summary count of annotations by type"""
        summary = {"button": 0, "input": 0, "radio": 0, "dropdown": 0}

        for annotation in annotations:
            tag = annotation.get("tag", "button")
            summary[tag] = summary.get(tag, 0) + 1

        return summary

    def render_grid_view(
        self, annotations_list: List[Tuple[str, List[Dict]]], cols: int = 2
    ):
        """Render multiple images with annotations in a grid layout"""

        for i in range(0, len(annotations_list), cols):
            columns = st.columns(cols)

            for j, col in enumerate(columns):
                if i + j < len(annotations_list):
                    image_name, annotations = annotations_list[i + j]

                    with col:
                        st.subheader(image_name)
                        fig = self.render_with_annotations(
                            annotations=annotations, width=400, height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show quick stats
                        summary = self._get_annotation_summary(annotations)
                        total_annotations = sum(summary.values())
                        st.write(f"**Total annotations:** {total_annotations}")


def create_annotation_heatmap(
    annotations: List[Dict], image_width: int, image_height: int
):
    """Create a heatmap showing annotation density across the image"""

    # Create a grid to track annotation density
    grid_size = 20
    grid_x = image_width // grid_size
    grid_y = image_height // grid_size

    heatmap_data = [[0 for _ in range(grid_x)] for _ in range(grid_y)]

    # Count annotations in each grid cell
    for annotation in annotations:
        bbox = annotation["bounding_box"]
        center_x = bbox["x"] + bbox["width"] // 2
        center_y = bbox["y"] + bbox["height"] // 2

        grid_col = min(int(center_x / grid_size), grid_x - 1)
        grid_row = min(int(center_y / grid_size), grid_y - 1)

        heatmap_data[grid_row][grid_col] += 1

    # Create heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(z=heatmap_data, colorscale="YlOrRd", showscale=True)
    )

    fig.update_layout(
        title="Annotation Density Heatmap",
        xaxis_title="X Position (grid)",
        yaxis_title="Y Position (grid)",
        width=400,
        height=300,
    )

    return fig
