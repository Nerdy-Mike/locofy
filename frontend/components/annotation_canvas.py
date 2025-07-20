import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

import streamlit as st


class AnnotationCanvas:
    """Interactive canvas for drawing bounding box annotations on images"""

    def __init__(self, image: Image.Image, annotations: List[Dict] = None):
        self.image = image
        self.annotations = annotations or []
        self.colors = {
            "button": "#FF0000",  # Red
            "input": "#0000FF",  # Blue
            "radio": "#00FF00",  # Green
            "dropdown": "#FFA500",  # Orange
        }

    def render_canvas(self, canvas_key: str = "annotation_canvas") -> Dict:
        """Render the interactive annotation canvas"""

        # Calculate canvas dimensions while maintaining aspect ratio
        max_width = 800
        max_height = 600

        img_width, img_height = self.image.size
        aspect_ratio = img_width / img_height

        if img_width > max_width:
            canvas_width = max_width
            canvas_height = int(max_width / aspect_ratio)
        else:
            canvas_width = img_width
            canvas_height = img_height

        if canvas_height > max_height:
            canvas_height = max_height
            canvas_width = int(max_height * aspect_ratio)

        # Scale factors for converting canvas coordinates to image coordinates
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height

        # Create background image with existing annotations
        background_image = self._create_background_with_annotations(
            canvas_width, canvas_height, scale_x, scale_y
        )

        # Drawing mode selection
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            drawing_mode = st.selectbox(
                "Drawing Mode",
                ["rect", "transform"],
                index=0,
                help="rect: Draw new rectangles, transform: Move/resize existing",
            )

        with col2:
            stroke_color = st.selectbox(
                "Component Type",
                list(self.colors.keys()),
                help="Select the type of UI component to annotate",
            )

        with col3:
            stroke_width = st.slider("Stroke Width", 1, 10, 3)

        # Create the drawable canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.1)",  # Transparent fill
            stroke_width=stroke_width,
            stroke_color=self.colors[stroke_color],
            background_color="#FFFFFF",
            background_image=background_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            point_display_radius=0,
            key=canvas_key,
            display_toolbar=True,
        )

        # Process canvas data
        result = {
            "canvas_data": canvas_result,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "new_annotations": [],
        }

        if canvas_result.json_data is not None:
            # Extract new rectangles from canvas
            objects = canvas_result.json_data["objects"]
            new_rects = [obj for obj in objects if obj["type"] == "rect"]

            for rect in new_rects:
                # Convert canvas coordinates back to image coordinates
                x = rect["left"] * scale_x
                y = rect["top"] * scale_y
                width = rect["width"] * scale_x
                height = rect["height"] * scale_y

                # Determine component type from stroke color
                component_type = self._color_to_component_type(rect["stroke"])

                result["new_annotations"].append(
                    {
                        "bounding_box": {
                            "x": round(x),
                            "y": round(y),
                            "width": round(width),
                            "height": round(height),
                        },
                        "tag": component_type,
                        "canvas_rect": rect,
                    }
                )

        return result

    def _create_background_with_annotations(
        self, canvas_width: int, canvas_height: int, scale_x: float, scale_y: float
    ) -> Image.Image:
        """Create background image with existing annotations drawn"""

        # Resize the original image to canvas size
        background = self.image.resize(
            (canvas_width, canvas_height), Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(background)

        # Draw existing annotations
        for annotation in self.annotations:
            bbox = annotation["bounding_box"]
            tag = annotation["tag"]
            color = self.colors.get(tag, "#000000")

            # Convert image coordinates to canvas coordinates
            x1 = bbox["x"] / scale_x
            y1 = bbox["y"] / scale_y
            x2 = (bbox["x"] + bbox["width"]) / scale_x
            y2 = (bbox["y"] + bbox["height"]) / scale_y

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label
            label = f"{tag}"
            draw.text((x1, y1 - 15), label, fill=color)

        return background

    def _color_to_component_type(self, color: str) -> str:
        """Convert stroke color back to component type"""
        color_to_type = {v: k for k, v in self.colors.items()}
        return color_to_type.get(color, "button")

    def render_annotation_list(self, annotations: List[Dict], on_delete_callback=None):
        """Render a list of existing annotations with edit/delete options"""

        if not annotations:
            st.info(
                "No annotations yet. Draw rectangles on the image above to create annotations."
            )
            return

        st.subheader(f"Annotations ({len(annotations)})")

        for i, annotation in enumerate(annotations):
            bbox = annotation["bounding_box"]
            tag = annotation["tag"]
            annotator = annotation.get("annotator", "Unknown")
            confidence = annotation.get("confidence")

            with st.expander(f"Annotation {i+1}: {tag.title()}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**Position:** ({bbox['x']}, {bbox['y']})")
                    st.write(f"**Size:** {bbox['width']} × {bbox['height']}")
                    st.write(f"**Area:** {bbox['width'] * bbox['height']} px²")

                with col2:
                    st.write(f"**Type:** {tag.title()}")
                    st.write(f"**Annotator:** {annotator}")
                    if confidence is not None:
                        st.write(f"**Confidence:** {confidence:.2f}")

                with col3:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{annotation['id']}",
                        help="Delete this annotation",
                    ):
                        if on_delete_callback:
                            on_delete_callback(annotation["id"])

                    if st.button(
                        "✏️ Edit",
                        key=f"edit_{annotation['id']}",
                        help="Edit this annotation",
                    ):
                        st.session_state[f"editing_{annotation['id']}"] = True

                # Edit form (if in edit mode)
                if st.session_state.get(f"editing_{annotation['id']}", False):
                    with st.form(f"edit_form_{annotation['id']}"):
                        st.write("**Edit Annotation**")

                        col_x, col_y = st.columns(2)
                        with col_x:
                            new_x = st.number_input(
                                "X",
                                value=bbox["x"],
                                min_value=0,
                                key=f"edit_x_{annotation['id']}",
                            )
                        with col_y:
                            new_y = st.number_input(
                                "Y",
                                value=bbox["y"],
                                min_value=0,
                                key=f"edit_y_{annotation['id']}",
                            )

                        col_w, col_h = st.columns(2)
                        with col_w:
                            new_width = st.number_input(
                                "Width",
                                value=bbox["width"],
                                min_value=1,
                                key=f"edit_w_{annotation['id']}",
                            )
                        with col_h:
                            new_height = st.number_input(
                                "Height",
                                value=bbox["height"],
                                min_value=1,
                                key=f"edit_h_{annotation['id']}",
                            )

                        new_tag = st.selectbox(
                            "Component Type",
                            ["button", "input", "radio", "dropdown"],
                            index=["button", "input", "radio", "dropdown"].index(tag),
                            key=f"edit_tag_{annotation['id']}",
                        )

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.form_submit_button("💾 Save Changes"):
                                # Return the updated annotation data
                                return {
                                    "action": "update",
                                    "annotation_id": annotation["id"],
                                    "updates": {
                                        "bounding_box": {
                                            "x": new_x,
                                            "y": new_y,
                                            "width": new_width,
                                            "height": new_height,
                                        },
                                        "tag": new_tag,
                                    },
                                }

                        with col_cancel:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state[f"editing_{annotation['id']}"] = False
                                st.rerun()

        return None


def create_annotation_summary(annotations: List[Dict]) -> Dict:
    """Create a summary of annotations by type"""
    summary = {"button": 0, "input": 0, "radio": 0, "dropdown": 0}

    for annotation in annotations:
        tag = annotation.get("tag", "button")
        summary[tag] = summary.get(tag, 0) + 1

    return summary


def validate_annotation_bounds(
    annotation: Dict, image_width: int, image_height: int
) -> bool:
    """Validate that annotation bounds are within image dimensions"""
    bbox = annotation["bounding_box"]

    # Check if annotation is within image bounds
    if (
        bbox["x"] < 0
        or bbox["y"] < 0
        or bbox["x"] + bbox["width"] > image_width
        or bbox["y"] + bbox["height"] > image_height
    ):
        return False

    # Check minimum size
    if bbox["width"] < 5 or bbox["height"] < 5:
        return False

    return True
