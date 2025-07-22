"""
Annotation Canvas Component

Implements the interactive annotation interface for drawing bounding boxes
and assigning tags to UI elements using streamlit-drawable-canvas.
"""

import io
import json
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st


class AnnotationSession:
    """Manages annotation session state as defined in DATAFLOW.md"""

    def __init__(self, image_id: str):
        self.image_id = image_id
        self.draft_annotations: List[Dict] = []
        self.is_drawing = False
        self.current_temp_box: Optional[Dict] = None

    def add_draft(self, bbox: Dict) -> str:
        """Add new draft annotation and return temp ID"""
        temp_id = f"temp_{len(self.draft_annotations) + 1}"
        draft = {
            "temp_id": temp_id,
            "bounding_box": bbox,
            "tag": None,
            "created_at": st.session_state.get("current_time", ""),
            "color": "#007bff",  # Draft color
        }
        self.draft_annotations.append(draft)
        return temp_id

    def assign_tag(self, temp_id: str, tag: str):
        """Assign tag to a draft annotation"""
        for draft in self.draft_annotations:
            if draft["temp_id"] == temp_id:
                draft["tag"] = tag
                draft["color"] = "#28a745"  # Tagged color
                break

    def remove_draft(self, temp_id: str):
        """Remove a draft annotation"""
        self.draft_annotations = [
            draft for draft in self.draft_annotations if draft["temp_id"] != temp_id
        ]

    def ready_to_save(self) -> bool:
        """Check if all drafts have tags assigned"""
        return len(self.draft_annotations) > 0 and all(
            draft["tag"] is not None for draft in self.draft_annotations
        )

    def get_annotations_for_api(self) -> List[Dict]:
        """Convert draft annotations to API format"""
        api_annotations = []
        for draft in self.draft_annotations:
            if draft["tag"]:  # Only include tagged annotations
                api_annotations.append(
                    {
                        "bounding_box": draft["bounding_box"],
                        "tag": draft["tag"],
                        "confidence": None,
                        "reasoning": None,
                    }
                )
        return api_annotations


def create_annotation_canvas(
    image_data: bytes,
    image_id: str,
    existing_annotations: List[Dict] = None,
    session_key: str = "annotation_session",
) -> AnnotationSession:
    """
    Create interactive annotation canvas using streamlit-drawable-canvas

    Args:
        image_data: Raw image bytes
        image_id: Unique image identifier
        existing_annotations: List of existing annotations to display
        session_key: Session state key for annotation session

    Returns:
        AnnotationSession: Current annotation session
    """

    # Initialize annotation session
    if session_key not in st.session_state:
        st.session_state[session_key] = AnnotationSession(image_id)

    session = st.session_state[session_key]
    existing_annotations = existing_annotations or []

    # Load and process image
    # Handle both bytes and BytesIO objects
    if isinstance(image_data, (io.BytesIO, io.BufferedReader)):
        # Reset position to beginning in case it was read before
        image_data.seek(0)
        image = Image.open(image_data)
    else:
        image = Image.open(io.BytesIO(image_data))
    img_width, img_height = image.size

    # Calculate display dimensions (maintain aspect ratio)
    max_width = 800
    max_height = 600

    scale_x = max_width / img_width
    scale_y = max_height / img_height
    scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down

    display_width = int(img_width * scale)
    display_height = int(img_height * scale)

    # Convert image for display
    if image.mode in ("RGBA", "LA", "P"):
        # Convert to RGB for better compatibility
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        image = rgb_image

    # Resize image for display
    display_image = image.resize(
        (display_width, display_height), Image.Resampling.LANCZOS
    )

    # Display instructions
    st.markdown("### 📝 Interactive Annotation Guide")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        **🎯 How to annotate:**
        1. **Draw rectangles** around UI elements by clicking and dragging
        2. **View your boxes** - they appear immediately in Python!
        3. **Assign tags** using the controls below the canvas
        4. **Save annotations** when all boxes are tagged
        """
        )

    with col2:
        st.info(
            f"""
        **📐 Image Info:**
        - Original: {img_width}×{img_height}px
        - Display: {display_width}×{display_height}px
        - Scale: {scale:.3f}x
        """
        )

    # Drawing tool selection
    st.markdown("### 🎨 Drawing Tools")

    tool_col1, tool_col2, tool_col3 = st.columns(3)

    with tool_col1:
        drawing_mode = st.selectbox(
            "Drawing Mode:",
            ["rect", "freedraw", "point", "transform"],
            index=0,
            help="Rectangle mode is recommended for UI element annotation",
        )

    with tool_col2:
        stroke_width = st.slider("Stroke Width:", 1, 10, 2)

    with tool_col3:
        stroke_color = st.color_picker("Stroke Color:", "#ff0000")

    # Canvas key for this specific image
    canvas_key = f"canvas_{image_id}"

    # Create the drawable canvas
    st.markdown("### 🖼️ Annotation Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.1)",  # Semi-transparent red fill
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#ffffff",
        background_image=display_image,
        update_streamlit=True,
        width=display_width,
        height=display_height,
        drawing_mode=drawing_mode,
        point_display_radius=5,
        key=canvas_key,
    )

    # Process drawn rectangles automatically
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        rectangles = [obj for obj in objects if obj["type"] == "rect"]

        # Convert canvas rectangles to annotations
        new_annotations = []
        for i, rect in enumerate(rectangles):
            # Get rectangle coordinates and scale back to original image size
            canvas_x = rect["left"]
            canvas_y = rect["top"]
            canvas_width = rect["width"] * rect["scaleX"]
            canvas_height = rect["height"] * rect["scaleY"]

            # Scale back to original image coordinates
            original_x = int(canvas_x / scale)
            original_y = int(canvas_y / scale)
            original_width = int(canvas_width / scale)
            original_height = int(canvas_height / scale)

            # Create annotation
            annotation = {
                "temp_id": f"canvas_{i+1}",
                "bounding_box": {
                    "x": original_x,
                    "y": original_y,
                    "width": original_width,
                    "height": original_height,
                },
                "tag": None,
                "created_at": str(st.session_state.get("current_time", "")),
                "color": "#007bff",
            }
            new_annotations.append(annotation)

        # Update session with new annotations (only if different)
        if len(new_annotations) != len(session.draft_annotations):
            session.draft_annotations = new_annotations

            if new_annotations:
                st.success(
                    f"🎉 **{len(new_annotations)} boxes drawn!** Ready for tagging below."
                )
            else:
                st.info("🎯 Draw rectangles around UI elements to create annotations.")

    # Display current annotations status
    st.markdown("---")
    st.markdown("### 📊 Annotation Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.metric("📦 Total Boxes", len(session.draft_annotations))

    with status_col2:
        tagged_count = len([ann for ann in session.draft_annotations if ann.get("tag")])
        st.metric("✅ Tagged", tagged_count)

    with status_col3:
        remaining = len(session.draft_annotations) - tagged_count
        st.metric("⏳ Remaining", remaining)

    # Show coordinates of drawn boxes
    if session.draft_annotations:
        with st.expander("🔍 View Box Coordinates", expanded=False):
            for i, ann in enumerate(session.draft_annotations):
                bbox = ann["bounding_box"]
                tag_status = f"🏷️ {ann['tag']}" if ann.get("tag") else "🔘 No tag"
                st.write(
                    f"**Box {i+1}:** ({bbox['x']}, {bbox['y']}) - {bbox['width']}×{bbox['height']}px - {tag_status}"
                )

    # Legend
    with st.expander("📖 Canvas Legend", expanded=False):
        st.markdown(
            """
        **🎨 Drawing Controls:**
        - **Rectangle Mode**: Click and drag to create bounding boxes
        - **Transform Mode**: Move and resize existing boxes
        - **Freedraw Mode**: Draw freehand annotations
        - **Point Mode**: Add point markers
        
        **🖱️ Canvas Controls:**
        - **Left Click + Drag**: Create rectangle (in rect mode)
        - **Click on Box**: Select/transform existing rectangle
        - **Double Click**: Delete selected element
        """
        )

    return session


def get_ui_element_config():
    """Get UI element configuration from JSON with enhanced fallback"""
    try:
        import json
        import os
        import sys

        # Try to load from the config file
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "utils", "annotation_config.json"
        )

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            ui_types = config.get("ui_element_types", {})
            tags = list(ui_types.keys())
            display_names = {
                tag: ui_types[tag].get("display_name", tag.title()) for tag in tags
            }

            return tags, display_names
        else:
            # Try the config loader as fallback
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
            from utils.annotation_config_loader import (
                get_ui_element_display_names,
                get_ui_element_list,
            )

            return get_ui_element_list(), get_ui_element_display_names()

    except (ImportError, FileNotFoundError, json.JSONDecodeError):
        # Enhanced fallback configuration with more UI element types
        tags = [
            "button",
            "input",
            "radio",
            "dropdown",
            "checkbox",
            "link",
            "image",
            "text",
            "toolbar",
            "menu",
            "slider",
            "tab",
        ]
        display_names = {
            "button": "Button",
            "input": "Input Field",
            "radio": "Radio Button",
            "dropdown": "Dropdown Menu",
            "checkbox": "Checkbox",
            "link": "Link",
            "image": "Image",
            "text": "Text",
            "toolbar": "Toolbar",
            "menu": "Menu",
            "slider": "Slider",
            "tab": "Tab",
        }
        return tags, display_names


def annotation_controls(
    session: AnnotationSession, ui_element_tags: List[str] = None
) -> Optional[str]:
    """
    Create annotation controls for tagging and saving

    Args:
        session: Current annotation session
        ui_element_tags: Available UI element tags

    Returns:
        str: Action taken ('save', 'clear', etc.) or None
    """
    # Load UI element configuration with enhanced descriptions
    if ui_element_tags is None:
        ui_element_tags, ui_display_names = get_ui_element_config()
    else:
        ui_display_names = {tag: tag.title() for tag in ui_element_tags}

    # Enhanced UI element descriptions for better user experience
    ui_descriptions = {
        "button": "Clickable buttons, submit buttons, action triggers",
        "input": "Text fields, search boxes, form inputs",
        "radio": "Radio buttons, single-choice options",
        "dropdown": "Select boxes, dropdowns, combo boxes",
        "checkbox": "Checkboxes, toggles, multi-choice options",
        "link": "Hyperlinks, navigation links, clickable text",
        "image": "Pictures, icons, logos, visual elements",
        "text": "Static text, labels, headings, paragraphs",
        "toolbar": "Toolbars, button groups, action bars",
        "menu": "Navigation menus, context menus",
        "slider": "Range sliders, progress bars",
        "tab": "Tab controls, page selectors",
    }

    # UI element icons for better visual identification
    ui_icons = {
        "button": "🔘",
        "input": "📝",
        "radio": "🔘",
        "dropdown": "📋",
        "checkbox": "☑️",
        "link": "🔗",
        "image": "🖼️",
        "text": "📄",
        "toolbar": "🔧",
        "menu": "📚",
        "slider": "🎚️",
        "tab": "📑",
    }

    # Show annotation summary with enhanced progress tracking
    total_boxes = len(session.draft_annotations)
    tagged_boxes = len([ann for ann in session.draft_annotations if ann.get("tag")])

    if total_boxes == 0:
        st.info(
            """
        📝 **No annotations yet!**
        
        **To get started:**
        1. **Draw rectangles** on the canvas above around UI elements
        2. **Boxes appear here automatically** - no sync needed!
        3. **Assign tags** to each box below
        4. **Save** when all boxes are tagged
        
        *Much simpler with streamlit-drawable-canvas! 🎉*
        """
        )
        return None

    # Enhanced progress indicator with visual feedback
    progress = tagged_boxes / total_boxes if total_boxes > 0 else 0
    progress_text = (
        f"Progress: {tagged_boxes}/{total_boxes} boxes tagged ({progress:.0%})"
    )
    st.progress(progress, text=progress_text)

    # Status messages with better styling
    if tagged_boxes == total_boxes:
        st.success("✅ All boxes are tagged and ready to save!")
    elif tagged_boxes > 0:
        st.warning(
            f"⚠️ {total_boxes - tagged_boxes} boxes still need tags before saving."
        )
    else:
        st.info("🏷️ Please assign UI element types to your drawn boxes below.")

    # Enhanced annotation manager section
    st.subheader("📦 Annotation Manager")

    # Quick stats
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("📦 Total Boxes", total_boxes)
    with stats_col2:
        st.metric("✅ Tagged", tagged_boxes)
    with stats_col3:
        st.metric("⏳ Remaining", total_boxes - tagged_boxes)
    with stats_col4:
        completion_pct = f"{progress:.0%}"
        st.metric("📈 Complete", completion_pct)

    st.markdown("---")

    # Enhanced tagging interface
    st.subheader("🏷️ UI Element Tagging")
    st.markdown("*Assign appropriate UI element types to each bounding box:*")

    action_taken = None

    # Enhanced tag assignment for each draft annotation
    for i, draft in enumerate(session.draft_annotations):
        # Create expandable section for each annotation
        bbox = draft["bounding_box"]
        current_tag = draft.get("tag")

        # Status indicator and summary
        if current_tag:
            status_emoji = "✅"
            status_color = "green"
            tag_display = f"**{ui_icons.get(current_tag, '📦')} {current_tag.upper()}**"
        else:
            status_emoji = "⏳"
            status_color = "orange"
            tag_display = "*(untagged)*"

        # Expandable annotation card
        with st.expander(
            f"{status_emoji} **Box {i+1}** - {tag_display}", expanded=not current_tag
        ):
            # Two column layout for annotation details
            detail_col1, detail_col2 = st.columns([2, 1])

            with detail_col1:
                # Box information
                st.markdown(f"**📍 Position:** `({bbox['x']}, {bbox['y']})`")
                st.markdown(
                    f"**📏 Dimensions:** `{bbox['width']} × {bbox['height']} pixels`"
                )

                # Area calculation
                area = bbox["width"] * bbox["height"]
                st.markdown(f"**📐 Area:** `{area:,} px²`")

            with detail_col2:
                # Visual preview (text-based)
                st.markdown("**📦 Box Preview:**")
                aspect_ratio = (
                    bbox["width"] / bbox["height"] if bbox["height"] > 0 else 1
                )
                if aspect_ratio > 2:
                    box_shape = "Wide rectangle 📏"
                elif aspect_ratio < 0.5:
                    box_shape = "Tall rectangle 📐"
                else:
                    box_shape = "Square/Rectangle ⬜"
                st.caption(box_shape)

            # Enhanced tag selection
            st.markdown("**🏷️ Select UI Element Type:**")

            tag_key = f"tag_{draft['temp_id']}"

            # Create enhanced display options with icons and descriptions
            tag_options = ["🔍 Select UI element type..."]
            for tag in ui_element_tags:
                icon = ui_icons.get(tag, "📦")
                display_name = ui_display_names.get(tag, tag.title())
                option_text = f"{icon} {display_name}"
                tag_options.append(option_text)

            # Find current index
            current_index = 0
            if current_tag:
                try:
                    tag_index = ui_element_tags.index(current_tag)
                    current_index = tag_index + 1
                except ValueError:
                    current_index = 0

            selected_option = st.selectbox(
                f"Choose type for Box {i+1}:",
                options=tag_options,
                index=current_index,
                key=tag_key,
                help="Select the most appropriate UI element type for this bounding box",
            )

            # Show description for selected tag
            if selected_option != "🔍 Select UI element type...":
                # Extract tag from the display option
                selected_tag = None
                for tag in ui_element_tags:
                    icon = ui_icons.get(tag, "📦")
                    display_name = ui_display_names.get(tag, tag.title())
                    if f"{icon} {display_name}" == selected_option:
                        selected_tag = tag
                        break

                if selected_tag:
                    description = ui_descriptions.get(selected_tag, "")
                    if description:
                        st.info(f"💡 **{selected_tag.title()}:** {description}")

                    # Update tag if changed
                    if selected_tag != current_tag:
                        session.assign_tag(draft["temp_id"], selected_tag)
                        st.rerun()

            # Action buttons for individual annotations
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button(
                    f"🗑️ Remove Box {i+1}",
                    key=f"remove_{draft['temp_id']}",
                    help="Remove this annotation completely",
                ):
                    session.remove_draft(draft["temp_id"])
                    st.rerun()

            with action_col2:
                if current_tag:
                    if st.button(
                        f"🔄 Reset Tag",
                        key=f"reset_{draft['temp_id']}",
                        help="Clear the tag assignment",
                    ):
                        session.assign_tag(draft["temp_id"], None)
                        st.rerun()

    st.markdown("---")

    # Enhanced save section with prominent styling
    st.markdown("## 💾 Save Annotations")

    # Save button status and requirements
    can_save = session.ready_to_save()
    save_col1, save_col2, save_col3 = st.columns([2, 1, 1])

    with save_col1:
        if can_save:
            st.success("✅ **Ready to save!** All boxes are properly tagged.")
        else:
            remaining = total_boxes - tagged_boxes
            st.warning(f"⚠️ **Cannot save yet.** {remaining} box(es) need tags.")

    with save_col2:
        # Save button with enhanced styling
        save_clicked = st.button(
            "💾 **Save All Annotations**",
            disabled=not can_save,
            help=(
                "Save all tagged annotations to the database"
                if can_save
                else "All boxes must be tagged before saving"
            ),
            type="primary" if can_save else "secondary",
            use_container_width=True,
        )
        if save_clicked:
            action_taken = "save"

    with save_col3:
        # Additional save options
        if st.button(
            "📤 Save & Continue",
            disabled=not can_save,
            help="Save annotations and continue adding more",
            use_container_width=True,
        ):
            action_taken = "save_continue"

    # Bulk action buttons
    st.markdown("### 🛠️ Bulk Actions")
    bulk_col1, bulk_col2, bulk_col3 = st.columns(3)

    with bulk_col1:
        if st.button(
            "🧹 Clear All Boxes",
            disabled=total_boxes == 0,
            help="Remove all draft annotations",
            use_container_width=True,
        ):
            session.draft_annotations = []
            action_taken = "clear"
            st.rerun()

    with bulk_col2:
        if st.button(
            "↶ Undo Last Box",
            disabled=total_boxes == 0,
            help="Remove the most recently added box",
            use_container_width=True,
        ):
            if session.draft_annotations:
                session.draft_annotations.pop()
                action_taken = "undo"
                st.rerun()

    with bulk_col3:
        if st.button(
            "🔄 Reset All Tags",
            disabled=tagged_boxes == 0,
            help="Clear all tag assignments but keep boxes",
            use_container_width=True,
        ):
            for draft in session.draft_annotations:
                draft["tag"] = None
            action_taken = "reset_tags"
            st.rerun()

    # Save preview section
    if can_save and total_boxes > 0:
        with st.expander("👀 Preview: What will be saved", expanded=False):
            st.markdown("**📋 Annotation Summary for Save:**")

            # Group by tag type
            tag_groups = {}
            for draft in session.draft_annotations:
                tag = draft.get("tag", "untagged")
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(draft)

            for tag, annotations in tag_groups.items():
                icon = ui_icons.get(tag, "📦")
                st.markdown(f"**{icon} {tag.title()}:** {len(annotations)} element(s)")

                # Show first few annotations as examples
                for i, ann in enumerate(annotations[:3]):
                    bbox = ann["bounding_box"]
                    st.caption(
                        f"  • Box at ({bbox['x']}, {bbox['y']}) - {bbox['width']}×{bbox['height']}px"
                    )

                if len(annotations) > 3:
                    st.caption(f"  • ... and {len(annotations) - 3} more")

    # Status messages for user guidance
    if not can_save and session.draft_annotations:
        untagged = [
            draft for draft in session.draft_annotations if not draft.get("tag")
        ]
        if len(untagged) == 1:
            st.info("💡 **Almost done!** Tag the remaining box to enable saving.")
        else:
            st.info(
                f"💡 **Progress update:** Tag {len(untagged)} more boxes to enable saving."
            )

    return action_taken
