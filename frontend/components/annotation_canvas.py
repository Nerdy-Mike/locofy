"""
Annotation Canvas Component

Implements the interactive annotation interface for drawing bounding boxes
and assigning tags to UI elements as designed in DATAFLOW.md.
"""

import base64
import json
import math
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import streamlit.components.v1 as components
from PIL import Image

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
    Create interactive annotation canvas

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
    image = Image.open(image_data)
    img_width, img_height = image.size

    # Convert image to base64 for HTML display (ensure PNG format)
    buffered = BytesIO()
    # Convert to RGB if necessary (in case of transparency issues)
    if image.mode in ("RGBA", "LA", "P"):
        # Convert RGBA/LA/P to RGB for better compatibility
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        rgb_image.save(buffered, format="JPEG", quality=95)
        img_data_url = (
            f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        )
    else:
        image.save(buffered, format="PNG")
        img_data_url = (
            f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        )

    # Calculate display dimensions (maintain aspect ratio)
    max_width = 800
    max_height = 600

    scale_x = max_width / img_width
    scale_y = max_height / img_height
    scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down

    display_width = int(img_width * scale)
    display_height = int(img_height * scale)

    # Create unique keys for this canvas instance
    canvas_key = f"canvas_annotations_{session_key}"
    sync_key = f"sync_{session_key}"

    # Initialize the annotation storage in session state if not exists
    if canvas_key not in st.session_state:
        st.session_state[canvas_key] = []

    # Create HTML/JS annotation interface
    annotation_html = f"""
    <style>
        .annotation-container {{
            position: relative;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            margin: 10px 0;
            background: #f8f9fa;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .annotation-canvas {{
            position: relative;
            cursor: crosshair;
            background-image: url('{img_data_url}');
            background-size: {display_width}px {display_height}px;
            background-repeat: no-repeat;
            background-position: top left;
            background-color: #f8f9fa;
            border: 1px solid #ccc;
        }}
        
        .annotation-box {{
            position: absolute;
            border: 2px solid;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            pointer-events: none;
            font-size: 11px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px black;
            padding: 2px 6px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: flex-start;
            min-height: 20px;
        }}
        
        .annotation-box .box-label {{
            background: rgba(0, 0, 0, 0.7);
            border-radius: 3px;
            padding: 2px 6px;
            margin: 2px;
            font-size: 10px;
            line-height: 1.2;
            max-width: calc(100% - 8px);
            word-wrap: break-word;
        }}
        
        .annotation-box .box-number {{
            background: rgba(0, 123, 255, 0.8);
            color: white;
        }}
        
        .annotation-box .box-tag {{
            background: rgba(40, 167, 69, 0.8);
            color: white;
            margin-top: 1px;
        }}
        
        .annotation-box.existing .box-tag {{
            background: rgba(108, 117, 125, 0.8);
        }}
        
        .annotation-box.draft {{
            border-color: #007bff;
            border-style: dashed;
        }}
        
        .annotation-box.tagged {{
            border-color: #28a745;
            border-style: solid;
        }}
        
        .annotation-box.existing {{
            border-color: #6c757d;
            border-style: solid;
        }}
        
        .annotation-box.temp {{
            border-color: #ffc107;
            border-style: dotted;
            background: rgba(255, 193, 7, 0.2);
        }}
        
        .annotation-instructions {{
            padding: 10px;
            background: #e9ecef;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        
        .sync-area {{
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
        }}
        
        .sync-button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }}
        
        .sync-button:hover {{
            background: #0056b3;
        }}
        
        .sync-button:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        
        .annotation-count {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }}
    </style>
    
    <div class="annotation-instructions">
        <strong>📝 How to annotate:</strong>
        <ol>
            <li><strong>Draw:</strong> Click and drag to create bounding boxes around UI elements</li>
            <li><strong>Sync:</strong> Click "Sync Annotations" to transfer boxes to the tagging controls</li>
            <li><strong>Tag:</strong> Assign UI element types (button, input, radio, dropdown) to each box</li>
            <li><strong>Save:</strong> Click "Save Annotations" when all boxes are tagged</li>
        </ol>
    </div>
    
    <div class="annotation-container">
        <div id="annotation-canvas" class="annotation-canvas" 
             style="width: {display_width}px; height: {display_height}px; min-height: 400px;">
        </div>
    </div>
    
    <div class="sync-area">
        <button id="sync-button" class="sync-button" onclick="syncAnnotations()">
            🔄 Sync Annotations
        </button>
        <button id="clear-button" class="sync-button" onclick="clearAllBoxes()">
            🧹 Clear Canvas
        </button>
        <span id="annotation-counter" class="annotation-count">0 boxes drawn</span>
        <input type="hidden" id="annotation-data" value="" />
    </div>
    
    <script>
        const canvas = document.getElementById('annotation-canvas');
        const annotationData = document.getElementById('annotation-data');
        const syncButton = document.getElementById('sync-button');
        const clearButton = document.getElementById('clear-button');
        const counter = document.getElementById('annotation-counter');
        const scale = {scale};
        const imgWidth = {img_width};
        const imgHeight = {img_height};
        
        let isDrawing = false;
        let startX = 0;
        let startY = 0;
        let currentBox = null;
        let drawnAnnotations = [];
        
        // Draft annotations from session (for display only)
        let draftAnnotations = {json.dumps(session.draft_annotations)};
        
        // Existing annotations  
        let existingAnnotations = {json.dumps(existing_annotations)};
        
        function updateCounter() {{
            counter.textContent = `${{drawnAnnotations.length}} boxes drawn`;
            syncButton.disabled = drawnAnnotations.length === 0;
            clearButton.disabled = drawnAnnotations.length === 0;
        }}
        
        function syncAnnotations() {{
            // Encode annotations as JSON and put in hidden input
            annotationData.value = JSON.stringify(drawnAnnotations);
            
            // Trigger a custom event that Streamlit can detect via a form submission
            const event = new CustomEvent('annotationsUpdated', {{
                detail: {{ annotations: drawnAnnotations }}
            }});
            document.dispatchEvent(event);
            
            // Visual feedback
            syncButton.textContent = '✅ Synced!';
            setTimeout(() => {{
                syncButton.textContent = '🔄 Sync Annotations';
            }}, 1500);
        }}
        
        function clearAllBoxes() {{
            drawnAnnotations = [];
            renderDrawnAnnotations();
            updateCounter();
            annotationData.value = '';
        }}
        
        function createBox(x, y, width, height, className, boxNumber = '', tagName = '', color = '#007bff') {{
            const box = document.createElement('div');
            box.className = 'annotation-box ' + className;
            box.style.left = x + 'px';
            box.style.top = y + 'px';
            box.style.width = width + 'px';
            box.style.height = height + 'px';
            box.style.borderColor = color;
            
            // Create labels
            if (boxNumber) {{
                const numberLabel = document.createElement('div');
                numberLabel.className = 'box-label box-number';
                numberLabel.textContent = boxNumber;
                box.appendChild(numberLabel);
            }}
            
            if (tagName) {{
                const tagLabel = document.createElement('div');
                tagLabel.className = 'box-label box-tag';
                tagLabel.textContent = tagName;
                box.appendChild(tagLabel);
            }}
            
            return box;
        }}
        
        function renderExistingAnnotations() {{
            existingAnnotations.forEach((ann, index) => {{
                const bbox = ann.bounding_box;
                const x = bbox.x * scale;
                const y = bbox.y * scale;
                const width = bbox.width * scale;
                const height = bbox.height * scale;
                
                const boxNumber = `#${{index + 1}}`;
                let displayName = ann.tag ? ann.tag.toUpperCase() : 'UNTAGGED';
                
                const box = createBox(x, y, width, height, 'existing', 
                                    boxNumber, displayName, '#6c757d');
                canvas.appendChild(box);
            }});
        }}
        
        function renderDraftAnnotations() {{
            // Clear existing draft boxes
            canvas.querySelectorAll('.annotation-box.draft, .annotation-box.tagged')
                .forEach(box => box.remove());
            
            draftAnnotations.forEach((draft, index) => {{
                const bbox = draft.bounding_box;
                const x = bbox.x * scale;
                const y = bbox.y * scale;
                const width = bbox.width * scale;
                const height = bbox.height * scale;
                
                const className = draft.tag ? 'tagged' : 'draft';
                const boxNumber = `Box ${{index + 1}}`;
                
                // Display tag name if available
                let displayName = '';
                if (draft.tag) {{
                    displayName = draft.tag.toUpperCase();
                }}
                
                const color = draft.tag ? '#28a745' : '#007bff';
                
                const box = createBox(x, y, width, height, className, 
                                    boxNumber, displayName, color);
                canvas.appendChild(box);
            }});
        }}
        
        function renderDrawnAnnotations() {{
            // Clear drawn boxes
            canvas.querySelectorAll('.annotation-box.temp-draft')
                .forEach(box => box.remove());
            
            drawnAnnotations.forEach((annotation, index) => {{
                const bbox = annotation.bounding_box;
                const x = bbox.x * scale;
                const y = bbox.y * scale;
                const width = bbox.width * scale;
                const height = bbox.height * scale;
                
                const boxNumber = `New ${{index + 1}}`;
                
                const box = createBox(x, y, width, height, 'temp-draft', 
                                    boxNumber, '', '#ff6b35');
                box.style.borderStyle = 'dashed';
                box.style.borderColor = '#ff6b35';
                canvas.appendChild(box);
            }});
        }}
        
        function getMousePos(e) {{
            const rect = canvas.getBoundingClientRect();
            return {{
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            }};
        }}
        
        function startDrawing(e) {{
            const pos = getMousePos(e);
            isDrawing = true;
            startX = pos.x;
            startY = pos.y;
            
            // Create temporary box
            currentBox = createBox(startX, startY, 0, 0, 'temp');
            canvas.appendChild(currentBox);
        }}
        
        function updateDrawing(e) {{
            if (!isDrawing || !currentBox) return;
            
            const pos = getMousePos(e);
            const x = Math.min(startX, pos.x);
            const y = Math.min(startY, pos.y);
            const width = Math.abs(pos.x - startX);
            const height = Math.abs(pos.y - startY);
            
            currentBox.style.left = x + 'px';
            currentBox.style.top = y + 'px';
            currentBox.style.width = width + 'px';
            currentBox.style.height = height + 'px';
        }}
        
        function finishDrawing(e) {{
            if (!isDrawing || !currentBox) return;
            
            const pos = getMousePos(e);
            const x = Math.min(startX, pos.x);
            const y = Math.min(startY, pos.y);
            const width = Math.abs(pos.x - startX);
            const height = Math.abs(pos.y - startY);
            
            // Remove temporary box
            currentBox.remove();
            currentBox = null;
            isDrawing = false;
            
            // Check minimum size
            if (width < 10 || height < 10) {{
                return;
            }}
            
            // Convert back to original image coordinates
            const originalX = x / scale;
            const originalY = y / scale;
            const originalWidth = width / scale;
            const originalHeight = height / scale;
            
            // Add to drawn annotations
            const newAnnotation = {{
                temp_id: 'drawn_' + (drawnAnnotations.length + 1),
                bounding_box: {{
                    x: Math.round(originalX),
                    y: Math.round(originalY),
                    width: Math.round(originalWidth),
                    height: Math.round(originalHeight)
                }},
                tag: null
            }};
            
            drawnAnnotations.push(newAnnotation);
            renderDrawnAnnotations();
            updateCounter();
        }}
        
        // Event listeners
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', updateDrawing);
        canvas.addEventListener('mouseup', finishDrawing);
        canvas.addEventListener('mouseleave', finishDrawing);
        
        // Initial render
        renderExistingAnnotations();
        renderDraftAnnotations();
        renderDrawnAnnotations();
        updateCounter();
    </script>
    """

    # Display image information and legend
    info_col1, info_col2 = st.columns([1, 1])

    with info_col1:
        st.write(
            f"**Image Info:** {img_width}×{img_height} pixels (original) → {display_width}×{display_height} pixels (display)"
        )
        st.write(f"**Scale Factor:** {scale:.3f}")

    with info_col2:
        st.write("**Legend:**")
        st.markdown(
            """
        🟠 **Orange (Dashed):** Newly drawn boxes (not synced)  
        🔵 **Blue (Dashed):** Draft boxes (synced, not tagged)  
        🟢 **Green (Solid):** Tagged boxes ready to save  
        ⚫ **Gray (Solid):** Existing saved annotations  
        🟡 **Yellow (Dotted):** Temporary drawing box  
        """
        )

    # Display the annotation canvas
    components.html(annotation_html, height=display_height + 250)

    # Add a manual sync button in Streamlit that can be used to transfer annotations
    st.markdown("---")

    # Create columns for sync controls
    sync_col1, sync_col2, sync_col3 = st.columns([1, 1, 1])

    with sync_col1:
        if st.button("🔄 Manual Sync", help="Transfer drawn boxes to tagging controls"):
            # This is a fallback sync method
            # In a real implementation, you'd need to implement a more sophisticated
            # method to get the annotations from JavaScript
            st.info("Use the 'Sync Annotations' button in the canvas area above")

    with sync_col2:
        drawn_count = len(st.session_state.get(canvas_key, []))
        draft_count = len(session.draft_annotations)
        st.metric("Canvas Boxes", drawn_count)

    with sync_col3:
        st.metric("Ready to Tag", draft_count)

    # Check if we need to transfer annotations from the hidden input
    # This is where we'd handle the JavaScript data if we had access to it
    # For now, we'll provide a simple method to manually add test annotations

    if st.button(
        "🧪 Add Test Box (for testing)", help="Add a test annotation to see controls"
    ):
        test_annotation = {
            "temp_id": f"test_{len(session.draft_annotations) + 1}",
            "bounding_box": {"x": 100, "y": 100, "width": 150, "height": 50},
            "tag": None,
        }
        session.draft_annotations.append(test_annotation)
        st.rerun()

    # Provide better instructions about the current limitation
    if len(session.draft_annotations) == 0:
        st.info(
            """
        📝 **Current Status**: The annotation canvas can draw boxes visually, but there's a technical limitation 
        in transferring the drawn boxes from JavaScript to Python in Streamlit. 
        
        **Workaround Options:**
        1. Click "🧪 Add Test Box" to see how the tagging controls work
        2. The visual drawing works - you can see boxes appear when you draw them
        3. Use the "Sync Annotations" button in the canvas (work in progress)
        
        **For Development**: This is a known limitation with Streamlit custom components communication.
        """
        )

    # Enhanced debugging information
    with st.expander("🔧 Debug Information"):
        st.write("**Session State Debug:**")
        st.write(f"- Session Key: `{session_key}`")
        st.write(f"- Canvas Key: `{canvas_key}`")
        st.write(f"- Draft Annotations Count: {len(session.draft_annotations)}")
        st.write(f"- Existing Annotations Count: {len(existing_annotations)}")

        if session.draft_annotations:
            st.write("**Current Draft Annotations:**")
            for i, draft in enumerate(session.draft_annotations):
                st.json(
                    {
                        f"Draft {i+1}": {
                            "temp_id": draft["temp_id"],
                            "bounding_box": draft["bounding_box"],
                            "tag": draft.get("tag"),
                        }
                    }
                )

        # Add manual annotation input for testing
        st.write("**Manual Annotation Entry (for testing):**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            x = st.number_input(
                "X", min_value=0, max_value=img_width, value=50, key="manual_x"
            )
        with col2:
            y = st.number_input(
                "Y", min_value=0, max_value=img_height, value=50, key="manual_y"
            )
        with col3:
            width = st.number_input(
                "Width",
                min_value=10,
                max_value=img_width,
                value=100,
                key="manual_width",
            )
        with col4:
            height = st.number_input(
                "Height",
                min_value=10,
                max_value=img_height,
                value=50,
                key="manual_height",
            )

        if st.button(
            "➕ Add Manual Annotation", help="Add annotation with specified coordinates"
        ):
            manual_annotation = {
                "temp_id": f"manual_{len(session.draft_annotations) + 1}",
                "bounding_box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                },
                "tag": None,
            }
            session.draft_annotations.append(manual_annotation)
            st.success(f"Added annotation at ({x}, {y}) with size {width}×{height}")
            st.rerun()

    return session


def get_ui_element_config():
    """Get UI element configuration from JSON"""
    try:
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        from utils.annotation_config_loader import (
            get_ui_element_display_names,
            get_ui_element_list,
        )

        return get_ui_element_list(), get_ui_element_display_names()
    except ImportError:
        # Fallback configuration
        tags = [
            "button",
            "input",
            "radio",
            "dropdown",
            "checkbox",
            "link",
            "image",
            "text",
        ]
        display_names = {tag: tag.title() for tag in tags}
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
    # Load UI element configuration
    if ui_element_tags is None:
        ui_element_tags, ui_display_names = get_ui_element_config()
    else:
        ui_display_names = {tag: tag.title() for tag in ui_element_tags}

    # Show annotation summary
    total_boxes = len(session.draft_annotations)
    tagged_boxes = len([ann for ann in session.draft_annotations if ann.get("tag")])

    if total_boxes == 0:
        st.info(
            """
        📝 **No annotations ready for tagging yet!**
        
        **To get started:**
        1. **Draw boxes** on the canvas above by clicking and dragging
        2. **Sync the boxes** using the "Sync Annotations" button in the canvas area
        3. **Or try the test button** "🧪 Add Test Box" to see how this interface works
        
        *Note: Due to Streamlit limitations, the JavaScript-to-Python communication is currently being improved.*
        """
        )
        return None

    # Progress indicator
    progress = tagged_boxes / total_boxes if total_boxes > 0 else 0
    st.progress(progress, text=f"Tagged: {tagged_boxes}/{total_boxes} boxes")

    if tagged_boxes == total_boxes:
        st.success("✅ All boxes are tagged! Ready to save.")
    elif tagged_boxes > 0:
        st.warning(f"⚠️ {total_boxes - tagged_boxes} boxes still need tags.")
    else:
        st.info("🏷️ Assign tags to your drawn boxes below.")

    st.subheader("🏷️ Tag Assignments")

    # Create columns for tag assignment
    cols = st.columns([3, 2, 1])

    action_taken = None

    # Tag assignment for each draft annotation
    for i, draft in enumerate(session.draft_annotations):
        with cols[0]:
            bbox = draft["bounding_box"]
            current_tag = draft.get("tag")

            # Show box info with status indicator
            if current_tag:
                status_emoji = "✅"
                tag_display = f"({current_tag.upper()})"
            else:
                status_emoji = "⚪"
                tag_display = "(untagged)"

            st.write(f"{status_emoji} **Box {i+1}** {tag_display}")
            st.caption(
                f"📍 Position: ({bbox['x']}, {bbox['y']}) | 📏 Size: {bbox['width']}×{bbox['height']}px"
            )

        with cols[1]:
            tag_key = f"tag_{draft['temp_id']}"

            # Create display options with descriptions
            tag_options = ["Select tag..."] + [
                f"{ui_display_names.get(tag, tag)} ({tag})" for tag in ui_element_tags
            ]

            # Find current index
            current_index = 0
            if current_tag:
                try:
                    tag_index = ui_element_tags.index(current_tag)
                    current_index = tag_index + 1
                except ValueError:
                    current_index = 0

            selected_option = st.selectbox(
                f"Tag for Box {i+1}",
                options=tag_options,
                index=current_index,
                key=tag_key,
                help=f"Assign a UI element type to Box {i+1}",
            )

            # Extract the actual tag from the display option
            if selected_option != "Select tag...":
                # Extract tag from "Display Name (tag)" format
                selected_tag = selected_option.split("(")[-1].rstrip(")")
            else:
                selected_tag = None

            if selected_tag and selected_tag != current_tag:
                session.assign_tag(draft["temp_id"], selected_tag)
                st.rerun()

        with cols[2]:
            if st.button(f"🗑️ Remove", key=f"remove_{draft['temp_id']}"):
                session.remove_draft(draft["temp_id"])
                st.rerun()

    st.markdown("---")

    # Action buttons
    button_cols = st.columns([1, 1, 1])

    with button_cols[0]:
        if st.button(
            "💾 Save Annotations",
            disabled=not session.ready_to_save(),
            help="All boxes must be tagged before saving",
        ):
            action_taken = "save"

    with button_cols[1]:
        if st.button("🧹 Clear All", disabled=len(session.draft_annotations) == 0):
            session.draft_annotations = []
            action_taken = "clear"
            st.rerun()

    with button_cols[2]:
        if st.button("↶ Undo Last", disabled=len(session.draft_annotations) == 0):
            if session.draft_annotations:
                session.draft_annotations.pop()
                action_taken = "undo"
                st.rerun()

    # Status messages
    if not session.ready_to_save() and session.draft_annotations:
        untagged = [
            draft for draft in session.draft_annotations if not draft.get("tag")
        ]
        st.warning(f"⚠️ {len(untagged)} box(es) still need tags before saving")

    return action_taken
