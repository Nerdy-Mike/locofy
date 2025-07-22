"""
Enhanced Annotation Renderer

Provides comprehensive annotation rendering capabilities for multiple sources:
- Manual annotations (ground truth)
- AI predictions (LLM/MCP generated)
- Draft annotations (user drawing)
- Imported annotations (external sources)

Features:
- Multi-source overlay rendering
- Customizable styling and colors
- Export capabilities (PNG, SVG, PDF)
- Interactive comparison views
- Statistical analysis overlays
"""

import base64
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from components.image_viewer import ImageViewer
from PIL import Image, ImageDraw, ImageFont

import streamlit as st


class AnnotationSource:
    """Represents a source of annotations with styling and metadata"""

    def __init__(
        self,
        name: str,
        annotations: List[Dict],
        color: str = "#FF0000",
        style: str = "solid",  # solid, dashed, dotted
        opacity: float = 0.8,
        show_labels: bool = True,
        show_confidence: bool = False,
        priority: int = 1,  # Higher priority renders on top
    ):
        self.name = name
        self.annotations = annotations
        self.color = color
        self.style = style
        self.opacity = opacity
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.priority = priority

    def get_annotation_count(self) -> int:
        """Get total number of annotations"""
        return len(self.annotations)

    def get_tag_summary(self) -> Dict[str, int]:
        """Get count of annotations by tag"""
        summary = {}
        for ann in self.annotations:
            tag = ann.get("tag", "unknown")
            summary[tag] = summary.get(tag, 0) + 1
        return summary


class MultiSourceAnnotationRenderer:
    """Enhanced renderer for multiple annotation sources"""

    # Predefined color schemes
    COLOR_SCHEMES = {
        "default": {
            "manual": "#28a745",  # Green for ground truth
            "ai_prediction": "#007bff",  # Blue for AI predictions
            "draft": "#ffc107",  # Yellow for drafts
            "imported": "#6c757d",  # Gray for imported
            "conflicted": "#dc3545",  # Red for conflicts
        },
        "accessible": {
            "manual": "#2E8B57",  # Sea Green
            "ai_prediction": "#4169E1",  # Royal Blue
            "draft": "#DAA520",  # Goldenrod
            "imported": "#708090",  # Slate Gray
            "conflicted": "#B22222",  # Fire Brick
        },
        "colorblind": {
            "manual": "#0173B2",  # Blue
            "ai_prediction": "#DE8F05",  # Orange
            "draft": "#CC78BC",  # Pink
            "imported": "#949494",  # Gray
            "conflicted": "#D55E00",  # Red-Orange
        },
    }

    # UI Component type colors for better visual distinction
    COMPONENT_COLORS = {
        "button": "#FF6B6B",  # Coral Red
        "input": "#4ECDC4",  # Teal
        "dropdown": "#45B7D1",  # Sky Blue
        "radio": "#96CEB4",  # Mint Green
        "checkbox": "#FFEAA7",  # Light Yellow
        "link": "#DDA0DD",  # Plum
        "image": "#F39C12",  # Orange
        "text": "#95A5A6",  # Gray
        "unknown": "#BDC3C7",  # Light Gray
    }

    def __init__(self, image: Union[Image.Image, bytes], color_scheme: str = "default"):
        if isinstance(image, bytes):
            self.image = Image.open(io.BytesIO(image))
        else:
            self.image = image

        self.width, self.height = self.image.size
        self.color_scheme = self.COLOR_SCHEMES.get(
            color_scheme, self.COLOR_SCHEMES["default"]
        )
        self.sources: List[AnnotationSource] = []

    def add_annotation_source(
        self,
        name: str,
        annotations: List[Dict],
        color: Optional[str] = None,
        style: str = "solid",
        opacity: float = 0.8,
        show_labels: bool = True,
        show_confidence: bool = False,
        priority: int = 1,
        use_component_colors: bool = True,
    ) -> None:
        """Add an annotation source"""

        # Auto-assign color if not provided
        if color is None:
            color = self.color_scheme.get(name.lower(), "#000000")

        # If using component colors, we'll assign colors per annotation based on UI type
        if use_component_colors:
            # Assign component-specific colors to each annotation
            for annotation in annotations:
                component_type = annotation.get("tag", "unknown")
                annotation["_component_color"] = self.COMPONENT_COLORS.get(
                    component_type, "#BDC3C7"
                )

        source = AnnotationSource(
            name=name,
            annotations=annotations,
            color=color,
            style=style,
            opacity=opacity,
            show_labels=show_labels,
            show_confidence=show_confidence,
            priority=priority,
        )

        self.sources.append(source)
        # Sort by priority (higher priority last, so they render on top)
        self.sources.sort(key=lambda x: x.priority)

    def render_plotly_interactive(
        self,
        width: int = 800,
        height: int = 600,
        show_legend: bool = True,
        enable_zoom: bool = True,
    ) -> go.Figure:
        """Render interactive visualization using Plotly"""

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
                opacity=1,
                layer="below",
            )
        )

        # Add annotations from each source
        for source in self.sources:
            for i, annotation in enumerate(source.annotations):
                self._add_plotly_annotation(
                    fig, annotation, source, f"{source.name}_{i}"
                )

        # Configure layout
        fig.update_layout(
            width=width,
            height=height,
            title=f"Annotations: {', '.join([s.name for s in self.sources])}",
            xaxis=dict(
                range=[0, self.width],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                range=[self.height, 0],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            showlegend=show_legend,
            margin=dict(l=20, r=20, t=60, b=20),
            dragmode="pan" if enable_zoom else False,
        )

        return fig

    def render_static_overlay(
        self,
        output_format: str = "PIL",  # PIL, bytes, matplotlib
        font_size: int = 12,
        box_thickness: int = 2,
        label_background: bool = True,
    ) -> Union[Image.Image, bytes, plt.Figure]:
        """Render static overlay on image"""

        if output_format == "matplotlib":
            return self._render_matplotlib(font_size, box_thickness)

        # Create PIL image with annotations
        img_copy = self.image.copy()
        draw = ImageDraw.Draw(img_copy)

        # Try to load a font
        try:
            # Attempt to use a system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except (OSError, IOError):
                # Fallback to default font
                font = ImageFont.load_default()

        # Draw annotations from each source
        for source in self.sources:
            for annotation in source.annotations:
                self._draw_pil_annotation(
                    draw, annotation, source, font, box_thickness, label_background
                )

        if output_format == "bytes":
            # Convert to bytes
            buffer = io.BytesIO()
            img_copy.save(buffer, format="PNG")
            return buffer.getvalue()

        return img_copy

    def render_comparison_grid(
        self,
        cols: int = 2,
        include_combined: bool = True,
        show_stats: bool = True,
    ) -> None:
        """Render comparison grid in Streamlit"""

        sources_to_show = self.sources.copy()
        if include_combined and len(self.sources) > 1:
            # Add combined view
            all_annotations = []
            for source in self.sources:
                all_annotations.extend(source.annotations)

            combined_source = AnnotationSource(
                "Combined View",
                all_annotations,
                color="#000000",
                show_labels=False,
            )
            sources_to_show.append(combined_source)

        # Calculate grid layout
        total_sources = len(sources_to_show)
        rows = (total_sources + cols - 1) // cols

        for row in range(rows):
            columns = st.columns(cols)

            for col_idx in range(cols):
                source_idx = row * cols + col_idx

                if source_idx < total_sources:
                    source = sources_to_show[source_idx]

                    with columns[col_idx]:
                        st.subheader(f"{source.name}")

                        # Create single-source renderer for this view
                        if source.name == "Combined View":
                            fig = self.render_plotly_interactive(
                                width=400, height=300, show_legend=False
                            )
                        else:
                            single_renderer = MultiSourceAnnotationRenderer(
                                self.image, "default"
                            )
                            single_renderer.add_annotation_source(
                                source.name,
                                source.annotations,
                                source.color,
                                source.style,
                                source.opacity,
                                source.show_labels,
                                source.show_confidence,
                            )
                            fig = single_renderer.render_plotly_interactive(
                                width=400, height=300, show_legend=False
                            )

                        st.plotly_chart(fig, use_container_width=True)

                        if show_stats:
                            # Show statistics
                            total_count = source.get_annotation_count()
                            st.metric("Total Annotations", total_count)

                            if total_count > 0:
                                tag_summary = source.get_tag_summary()
                                with st.expander("Tag Breakdown"):
                                    for tag, count in tag_summary.items():
                                        st.write(f"• {tag.title()}: {count}")

    def export_rendered_image(
        self,
        filename: str,
        format: str = "PNG",
        include_metadata: bool = True,
    ) -> bytes:
        """Export rendered image with annotations"""

        rendered_image = self.render_static_overlay(output_format="PIL")

        # Save to bytes buffer
        buffer = io.BytesIO()

        if include_metadata:
            # Add metadata to image
            metadata = {
                "annotation_sources": len(self.sources),
                "total_annotations": sum(
                    s.get_annotation_count() for s in self.sources
                ),
                "created_at": datetime.now().isoformat(),
                "image_size": f"{self.width}x{self.height}",
            }

            # Add metadata as PNG text fields (for PNG format)
            if format.upper() == "PNG":
                from PIL.PngImagePlugin import PngInfo

                pnginfo = PngInfo()
                for key, value in metadata.items():
                    pnginfo.add_text(key, str(value))
                rendered_image.save(buffer, format=format, pnginfo=pnginfo)
            else:
                rendered_image.save(buffer, format=format)
        else:
            rendered_image.save(buffer, format=format)

        return buffer.getvalue()

    def generate_annotation_report(self) -> Dict:
        """Generate comprehensive annotation report"""

        report = {
            "image_info": {
                "width": self.width,
                "height": self.height,
                "total_pixels": self.width * self.height,
            },
            "sources": [],
            "summary": {
                "total_sources": len(self.sources),
                "total_annotations": 0,
                "coverage_percentage": 0.0,
                "tag_distribution": {},
            },
        }

        total_annotations = 0
        covered_pixels = 0
        all_tags = {}

        for source in self.sources:
            source_info = {
                "name": source.name,
                "count": source.get_annotation_count(),
                "tags": source.get_tag_summary(),
                "average_confidence": 0.0,
                "coverage_pixels": 0,
            }

            # Calculate coverage and confidence
            confidences = []
            for annotation in source.annotations:
                bbox = annotation["bounding_box"]
                area = bbox["width"] * bbox["height"]
                source_info["coverage_pixels"] += area
                covered_pixels += area

                if "confidence" in annotation:
                    confidences.append(annotation["confidence"])

                # Update global tag distribution
                tag = annotation.get("tag", "unknown")
                all_tags[tag] = all_tags.get(tag, 0) + 1

            if confidences:
                source_info["average_confidence"] = sum(confidences) / len(confidences)

            total_annotations += source_info["count"]
            report["sources"].append(source_info)

        # Update summary
        report["summary"]["total_annotations"] = total_annotations
        report["summary"]["coverage_percentage"] = (
            covered_pixels / report["image_info"]["total_pixels"]
        ) * 100
        report["summary"]["tag_distribution"] = all_tags

        return report

    def _add_plotly_annotation(
        self, fig: go.Figure, annotation: Dict, source: AnnotationSource, trace_id: str
    ):
        """Add annotation to Plotly figure"""

        bbox = annotation["bounding_box"]
        tag = annotation.get("tag", "unknown")
        confidence = annotation.get("confidence")

        # Use component-specific color if available, otherwise use source color
        component_color = annotation.get("_component_color", source.color)

        # Determine line style
        line_style = {"color": component_color, "width": 3}  # Thicker border
        if source.style == "dashed":
            line_style["dash"] = "dash"
        elif source.style == "dotted":
            line_style["dash"] = "dot"

        # Add rectangle with more prominent background
        fig.add_shape(
            type="rect",
            x0=bbox["x"],
            y0=bbox["y"],
            x1=bbox["x"] + bbox["width"],
            y1=bbox["y"] + bbox["height"],
            line=line_style,
            fillcolor=component_color,
            opacity=0.4,  # More prominent background
            name=source.name,
        )

        # Add label if requested
        if source.show_labels:
            label_text = f"{tag.upper()}"
            if source.show_confidence and confidence is not None:
                label_text += f" ({confidence:.2f})"

            # Position label
            label_y = (
                bbox["y"] - 15 if bbox["y"] > 20 else bbox["y"] + bbox["height"] + 15
            )

            fig.add_annotation(
                x=bbox["x"] + bbox["width"] / 2,
                y=label_y,
                text=label_text,
                showarrow=False,
                font=dict(
                    color=component_color, size=12, family="Arial Black"
                ),  # Bolder font
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=component_color,
                borderwidth=2,
            )

    def _draw_pil_annotation(
        self,
        draw: ImageDraw.Draw,
        annotation: Dict,
        source: AnnotationSource,
        font,
        thickness: int,
        label_background: bool,
    ):
        """Draw annotation using PIL"""

        bbox = annotation["bounding_box"]
        x, y = bbox["x"], bbox["y"]
        x2, y2 = x + bbox["width"], y + bbox["height"]

        # Draw rectangle
        color = source.color
        if source.style == "dashed":
            # Simulate dashed line by drawing segments
            self._draw_dashed_rectangle(draw, x, y, x2, y2, color, thickness)
        else:
            draw.rectangle([x, y, x2, y2], outline=color, width=thickness)

        # Draw label if requested
        if source.show_labels:
            tag = annotation.get("tag", "unknown")
            confidence = annotation.get("confidence")

            label_text = f"{tag}"
            if source.show_confidence and confidence is not None:
                label_text += f" ({confidence:.2f})"

            # Position label
            label_y = y - 20 if y > 20 else y2 + 5

            if label_background:
                # Get text size
                bbox_text = draw.textbbox((x, label_y), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                # Draw background
                draw.rectangle(
                    [x, label_y, x + text_width + 4, label_y + text_height + 4],
                    fill="white",
                    outline=color,
                    width=1,
                )

            # Draw text
            draw.text((x + 2, label_y + 2), label_text, fill=color, font=font)

    def _draw_dashed_rectangle(
        self, draw: ImageDraw.Draw, x1, y1, x2, y2, color, thickness
    ):
        """Draw dashed rectangle"""
        dash_length = 10

        # Top edge
        for x in range(x1, x2, dash_length * 2):
            draw.line(
                [x, y1, min(x + dash_length, x2), y1], fill=color, width=thickness
            )

        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            draw.line(
                [x, y2, min(x + dash_length, x2), y2], fill=color, width=thickness
            )

        # Left edge
        for y in range(y1, y2, dash_length * 2):
            draw.line(
                [x1, y, x1, min(y + dash_length, y2)], fill=color, width=thickness
            )

        # Right edge
        for y in range(y1, y2, dash_length * 2):
            draw.line(
                [x2, y, x2, min(y + dash_length, y2)], fill=color, width=thickness
            )

    def _render_matplotlib(self, font_size: int, box_thickness: int) -> plt.Figure:
        """Render using matplotlib"""

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(self.image)

        # Add annotations from each source
        for source in self.sources:
            for annotation in source.annotations:
                bbox = annotation["bounding_box"]

                # Create rectangle
                linestyle = "-" if source.style == "solid" else "--"
                rect = patches.Rectangle(
                    (bbox["x"], bbox["y"]),
                    bbox["width"],
                    bbox["height"],
                    linewidth=box_thickness,
                    edgecolor=source.color,
                    facecolor=source.color,
                    alpha=source.opacity * 0.2,
                    linestyle=linestyle,
                    label=source.name,
                )
                ax.add_patch(rect)

                # Add label if requested
                if source.show_labels:
                    tag = annotation.get("tag", "unknown")
                    confidence = annotation.get("confidence")

                    label_text = f"{tag}"
                    if source.show_confidence and confidence is not None:
                        label_text += f" ({confidence:.2f})"

                    ax.text(
                        bbox["x"],
                        bbox["y"] - 5,
                        label_text,
                        fontsize=font_size,
                        color=source.color,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                    )

        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.axis("off")

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove duplicate labels
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()
        return fig


def create_streamlit_annotation_interface(
    image_data: bytes,
    image_id: str,
    manual_annotations: List[Dict] = None,
    ai_predictions: List[Dict] = None,
    draft_annotations: List[Dict] = None,
    imported_annotations: List[Dict] = None,
) -> None:
    """Create comprehensive annotation interface in Streamlit"""

    renderer = MultiSourceAnnotationRenderer(image_data)

    # Control panel
    st.subheader("🎨 Annotation Visualization Controls")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        view_mode = st.selectbox(
            "View Mode",
            ["Interactive", "Comparison Grid", "Static Overlay"],
            help="Choose how to display annotations",
        )

    with col2:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["default", "accessible", "colorblind"],
            help="Choose color scheme for annotations",
        )
        renderer.color_scheme = renderer.COLOR_SCHEMES[color_scheme]

    with col3:
        use_component_colors = st.checkbox(
            "Component Colors",
            value=True,
            help="Use different colors for different UI component types (button, input, dropdown, etc.)",
        )

    with col4:
        show_labels = st.checkbox(
            "Show Labels",
            value=False,
            help="Show component type labels and confidence scores on annotations",
        )

    # Additional row for statistics
    show_stats = st.checkbox("Show Statistics", value=True)

    # Show component color legend if enabled
    if use_component_colors:
        st.markdown("**🎨 Component Color Legend:**")
        legend_cols = st.columns(len(renderer.COMPONENT_COLORS))
        for i, (component, color) in enumerate(renderer.COMPONENT_COLORS.items()):
            with legend_cols[i % len(legend_cols)]:
                st.markdown(
                    f'<span style="color: {color}; font-weight: bold;">●</span> {component.title()}',
                    unsafe_allow_html=True,
                )

    # Update annotation sources with component coloring preference
    renderer.sources = []  # Clear existing sources

    # Re-add annotation sources with updated settings
    if manual_annotations:
        renderer.add_annotation_source(
            "Manual",
            manual_annotations,
            style="solid",
            show_labels=show_labels,
            priority=3,
            use_component_colors=use_component_colors,
        )

    if ai_predictions:
        renderer.add_annotation_source(
            "AI_Prediction",
            ai_predictions,
            style="dashed",
            show_labels=show_labels,
            show_confidence=show_labels,
            priority=2,
            use_component_colors=use_component_colors,
        )

    if draft_annotations:
        renderer.add_annotation_source(
            "Draft",
            draft_annotations,
            style="dotted",
            show_labels=show_labels,
            priority=1,
            use_component_colors=use_component_colors,
        )

    if imported_annotations:
        renderer.add_annotation_source(
            "Imported",
            imported_annotations,
            style="solid",
            opacity=0.6,
            show_labels=show_labels,
            priority=4,
            use_component_colors=use_component_colors,
        )

    # Render based on selected mode
    if view_mode == "Interactive":
        st.subheader("🔍 Interactive View")

        enable_zoom = st.checkbox("Enable Zoom/Pan", value=True)
        show_legend = st.checkbox("Show Legend", value=True)

        fig = renderer.render_plotly_interactive(
            width=800,
            height=600,
            show_legend=show_legend,
            enable_zoom=enable_zoom,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Comparison Grid":
        st.subheader("📊 Comparison Grid")

        cols = st.slider("Grid Columns", 1, 4, 2)
        include_combined = st.checkbox("Include Combined View", value=True)

        renderer.render_comparison_grid(
            cols=cols,
            include_combined=include_combined,
            show_stats=show_stats,
        )

    elif view_mode == "Static Overlay":
        st.subheader("🖼️ Static Overlay")

        col1, col2, col3 = st.columns(3)
        with col1:
            font_size = st.slider("Font Size", 8, 24, 12)
        with col2:
            box_thickness = st.slider("Box Thickness", 1, 5, 2)
        with col3:
            label_background = st.checkbox("Label Background", value=True)

        rendered_image = renderer.render_static_overlay(
            output_format="PIL",
            font_size=font_size,
            box_thickness=box_thickness,
            label_background=label_background,
        )

        st.image(rendered_image, use_container_width=True)

        # Export options
        st.subheader("💾 Export Options")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            export_format = st.selectbox("Export Format", ["PNG", "JPEG", "PDF"])
            include_metadata = st.checkbox("Include Metadata", value=True)

        with export_col2:
            if st.button("📥 Download Rendered Image"):
                exported_data = renderer.export_rendered_image(
                    filename=f"annotated_{image_id}",
                    format=export_format,
                    include_metadata=include_metadata,
                )

                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=f"annotated_{image_id}.{export_format.lower()}",
                    mime=f"image/{export_format.lower()}",
                )

    # Show annotation report
    if show_stats:
        st.subheader("📈 Annotation Report")

        report = renderer.generate_annotation_report()

        # Summary metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        with metrics_col1:
            st.metric("Total Sources", report["summary"]["total_sources"])

        with metrics_col2:
            st.metric("Total Annotations", report["summary"]["total_annotations"])

        with metrics_col3:
            st.metric("Coverage %", f"{report['summary']['coverage_percentage']:.1f}%")

        with metrics_col4:
            st.metric(
                "Image Size",
                f"{report['image_info']['width']}×{report['image_info']['height']}",
            )

        # Detailed breakdown
        with st.expander("📋 Detailed Report"):
            st.json(report)
