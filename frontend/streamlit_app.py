# Import necessary libraries
import io
import json
import os
import sys
from pathlib import Path

from PIL import Image

import streamlit as st

# Configure Streamlit page FIRST - must be the very first Streamlit command
st.set_page_config(
    page_title="UI Component Labeling Tool",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add the frontend directory and parent directory to Python path for imports
# This ensures imports work in both local development and Docker environments
frontend_dir = Path(__file__).parent
project_root = frontend_dir.parent
sys.path.insert(0, str(frontend_dir))
sys.path.insert(0, str(project_root))

# Try multiple import paths for API client to handle different environments
api_client = None
import_success = ""
import_error = None

try:
    from utils.api_client import UILabelingAPIClient

    import_success = "Local import successful"
except ImportError as e1:
    try:
        from frontend.utils.api_client import UILabelingAPIClient

        import_success = "Frontend import successful"
    except ImportError as e2:
        import_error = {
            "error1": str(e1),
            "error2": str(e2),
            "cwd": os.getcwd(),
            "python_path": sys.path,
            "frontend_dir": str(frontend_dir),
            "project_root": str(project_root),
        }

# Handle import error after page config
if import_error:
    st.error("❌ Could not import UILabelingAPIClient. Please check the setup.")
    st.error(f"Error 1: {import_error['error1']}")
    st.error(f"Error 2: {import_error['error2']}")
    st.error(f"Current working directory: {import_error['cwd']}")
    st.error(f"Python path: {import_error['python_path']}")
    st.error(f"Frontend directory: {import_error['frontend_dir']}")
    st.error(f"Project root: {import_error['project_root']}")
    st.stop()

# Add diagnostic information in development
if os.getenv("ENVIRONMENT") == "development":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Debug Info**")
    st.sidebar.write(f"Import: {import_success}")
    st.sidebar.write(f"CWD: {os.getcwd()}")


# Initialize API client
@st.cache_resource
def get_api_client():
    """Get cached API client instance"""
    return UILabelingAPIClient()


api_client = get_api_client()

# Sidebar navigation
st.sidebar.title("🎨 UI Component Labeling")

pages = {
    "🏠 Image Management": "image_management",
    "📝 Annotation Tool": "annotation_tool",
    "🤖 AI Predictions": "ai_predictions",
    "🎨 Enhanced Annotation Viewer": "enhanced_annotation_viewer",
    "📊 Analytics & Export": "analytics",
}

page = st.sidebar.selectbox("Navigate to", list(pages.keys()))

# Display API connection status
with st.sidebar:
    st.subheader("🔌 Connection Status")
    try:
        health = api_client.get_health()
        if health.get("status") == "healthy":
            st.success("✅ API Connected")

            # Show service status
            services = health.get("services", {})
            storage_status = services.get("storage", "unknown")
            llm_status = services.get("llm", "unknown")

            st.write(f"**Storage:** {storage_status}")
            st.write(f"**LLM:** {llm_status}")

            # Show basic stats if available
            storage_stats = health.get("storage_stats", {})
            if storage_stats and "error" not in storage_stats:
                st.write(f"**Images:** {storage_stats.get('total_images', 0)}")
                st.write(
                    f"**Storage:** {storage_stats.get('total_storage_mb', 0):.1f} MB"
                )
        else:
            st.error("❌ API Unavailable")

    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Phase 1.1 Implementation**")
st.sidebar.markdown(
    "✅ Enhanced Upload Validation  \n✅ Comprehensive Error Handling  \n✅ Duplicate Detection  \n✅ Image Metadata Validation"
)

# === PAGE: IMAGE MANAGEMENT ===
if page == "🏠 Image Management":
    st.header("📁 Image Management")
    st.markdown("Upload and manage UI screenshots for component labeling.")

    # Create main layout
    upload_col, gallery_col = st.columns([1, 2])

    with upload_col:
        st.subheader("📤 Upload New Image")

        # File uploader with enhanced validation info
        st.markdown(
            """
        **Supported formats:** PNG, JPG, JPEG, GIF, BMP  
        **File size:** 1KB - 10MB  
        **Minimum resolution:** 100x100 pixels  
        """
        )

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            help="Upload a UI screenshot or design image for labeling",
        )

        if uploaded_file is not None:
            # Show preview and file info
            st.subheader("📋 File Preview")

            # Get file content once at the beginning
            file_content = uploaded_file.getvalue()

            try:
                # Display image preview using BytesIO to avoid moving file pointer
                image = Image.open(io.BytesIO(file_content))
                st.image(
                    image,
                    caption=f"Preview: {uploaded_file.name}",
                    use_column_width=True,
                )

                # Show file information
                file_info = {
                    "Filename": uploaded_file.name,
                    "Size": f"{len(file_content) / 1024:.1f} KB",
                    "Dimensions": f"{image.width} x {image.height}",
                    "Format": image.format,
                    "Mode": image.mode,
                }

                for key, value in file_info.items():
                    st.write(f"**{key}:** {value}")

                # Check for potential issues and show warnings
                warnings = []

                # Size warnings
                if len(file_content) > 5 * 1024 * 1024:  # 5MB
                    warnings.append("Large file size may slow processing")

                # Dimension warnings
                if image.width < 200 or image.height < 200:
                    warnings.append("Low resolution may affect annotation accuracy")

                aspect_ratio = image.width / image.height
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    warnings.append("Unusual aspect ratio detected")

                # Show warnings if any
                if warnings:
                    st.warning(
                        "⚠️ **Potential Issues:**\n"
                        + "\n".join([f"• {w}" for w in warnings])
                    )

            except Exception as e:
                st.error(f"❌ Could not preview image: {str(e)}")

            st.markdown("---")

            # Upload button and processing
            if st.button("🚀 Upload Image", type="primary", use_container_width=True):
                with st.spinner("🔍 Validating and uploading image..."):
                    try:
                        # Use the file_content we already read
                        response = api_client.upload_image(
                            file_content, uploaded_file.name
                        )

                        # Success handling
                        st.success("✅ **Upload Successful!**")

                        # Display upload results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Image ID", response["id"][:8] + "...")
                            st.metric(
                                "File Size", f"{response['file_size'] / 1024:.1f} KB"
                            )

                        with col2:
                            st.metric(
                                "Dimensions",
                                f"{response['width']}x{response['height']}",
                            )
                            st.metric("Format", response["format"])

                        # Show validation info if available
                        if (
                            "validation_info" in response
                            and response["validation_info"]
                        ):
                            validation_info = response["validation_info"]
                            with st.expander("🔍 Validation Details"):
                                st.write(
                                    f"**Checksum:** {validation_info['checksum'][:16]}..."
                                )
                                st.write(
                                    f"**Original Filename:** {validation_info['original_filename']}"
                                )
                                st.write(
                                    f"**Sanitized Filename:** {validation_info['sanitized_filename']}"
                                )
                                st.write(
                                    f"**Validation Time:** {validation_info['validation_timestamp']}"
                                )

                        # Auto-refresh to show in gallery
                        st.rerun()

                    except Exception as e:
                        # Enhanced error handling with specific error types
                        error_msg = str(e)

                        if "duplicate_image" in error_msg:
                            st.error(
                                "🔄 **Duplicate Image Detected**\n\nThis image has already been uploaded to the system."
                            )

                        elif "unsupported_file_type" in error_msg:
                            st.error(
                                "📄 **Unsupported File Type**\n\nPlease use PNG, JPG, JPEG, GIF, or BMP format."
                            )

                        elif "file_too_large" in error_msg:
                            st.error(
                                "📏 **File Too Large**\n\nMaximum file size is 10MB. Please compress your image."
                            )

                        elif "file_too_small" in error_msg:
                            st.error(
                                "📏 **File Too Small**\n\nMinimum file size is 1KB. The file may be corrupted."
                            )

                        elif "validation_failed" in error_msg:
                            st.error(
                                "❌ **Validation Failed**\n\nThe image failed quality validation. Please check the format and try again."
                            )

                        elif "invalid_filename" in error_msg:
                            st.error(
                                "📝 **Invalid Filename**\n\nPlease use a valid filename without special characters."
                            )

                        else:
                            st.error(f"❌ **Upload Failed**\n\n{error_msg}")

                        # Show troubleshooting tips
                        with st.expander("💡 Troubleshooting Tips"):
                            st.markdown(
                                """
                            **Common Solutions:**
                            - Ensure image is a valid UI screenshot
                            - Check file format (PNG, JPG, GIF, BMP)
                            - Verify file size is between 1KB and 10MB
                            - Try renaming the file if filename issues occur
                            - Ensure minimum resolution of 100x100 pixels
                            """
                            )

    with gallery_col:
        st.subheader("🖼️ Image Gallery")

        try:
            images = api_client.list_images()

            if not images:
                st.info(
                    "📥 **No images uploaded yet**\n\nUpload your first UI screenshot to get started!"
                )
            else:
                # Gallery statistics
                total_images = len(images)
                total_annotations = sum(
                    img.get("annotation_count", 0) for img in images
                )
                total_predictions = sum(
                    1 for img in images if img.get("has_ai_predictions", False)
                )

                # Show stats
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("📸 Total Images", total_images)
                with stat_col2:
                    st.metric("📝 Annotations", total_annotations)
                with stat_col3:
                    st.metric("🤖 AI Predictions", total_predictions)

                st.markdown("---")

                # Search and filter options
                search_col, filter_col = st.columns([2, 1])

                with search_col:
                    search_term = st.text_input(
                        "🔍 Search images", placeholder="Search by filename..."
                    )

                with filter_col:
                    format_filter = st.selectbox(
                        "Format Filter",
                        options=["All"]
                        + list(set(img.get("format", "UNKNOWN") for img in images)),
                    )

                # Filter images based on search and format
                filtered_images = images

                if search_term:
                    filtered_images = [
                        img
                        for img in filtered_images
                        if search_term.lower() in img.get("filename", "").lower()
                    ]

                if format_filter != "All":
                    filtered_images = [
                        img
                        for img in filtered_images
                        if img.get("format") == format_filter
                    ]

                st.write(f"Showing {len(filtered_images)} of {total_images} images")

                # Create image grid
                cols_per_row = 2
                for i in range(0, len(filtered_images), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j, col in enumerate(cols):
                        if i + j < len(filtered_images):
                            img_data = filtered_images[i + j]

                            with col:
                                try:
                                    # Get image thumbnail
                                    img_content = api_client.get_image_file(
                                        img_data["id"]
                                    )
                                    image = Image.open(io.BytesIO(img_content))

                                    # Display thumbnail with consistent sizing
                                    thumbnail_size = (300, 200)
                                    image.thumbnail(
                                        thumbnail_size, Image.Resampling.LANCZOS
                                    )
                                    st.image(image, use_column_width=True)

                                    # Image information card - simplified layout
                                    st.markdown(f"**{img_data['filename']}**")

                                    # Basic info in simple text format
                                    st.write(
                                        f"📐 {img_data['width']}×{img_data['height']} • 📁 {img_data.get('format', 'Unknown')}"
                                    )
                                    st.write(
                                        f"📝 {img_data['annotation_count']} annotations"
                                    )

                                    ai_status = (
                                        "✅"
                                        if img_data.get("has_ai_predictions")
                                        else "❌"
                                    )
                                    st.write(f"🤖 {ai_status} AI predictions")

                                    # File size and upload date
                                    file_size_mb = img_data.get("file_size", 0) / (
                                        1024 * 1024
                                    )
                                    st.write(f"💾 {file_size_mb:.2f} MB")

                                    # Upload timestamp
                                    upload_time = img_data.get("upload_time", "")
                                    if upload_time:
                                        from datetime import datetime

                                        try:
                                            dt = datetime.fromisoformat(
                                                upload_time.replace("Z", "+00:00")
                                            )
                                            st.write(
                                                f"📅 {dt.strftime('%Y-%m-%d %H:%M')}"
                                            )
                                        except:
                                            st.write(f"📅 {upload_time[:10]}")

                                    # Action buttons - vertical layout to avoid nesting
                                    if st.button(
                                        "✏️ Edit",
                                        key=f"edit_{img_data['id']}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.selected_image_id = img_data[
                                            "id"
                                        ]
                                        st.session_state.page = "annotation_tool"
                                        st.rerun()

                                    if st.button(
                                        "🔍 View",
                                        key=f"view_{img_data['id']}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.selected_image_id = img_data[
                                            "id"
                                        ]
                                        # Use a simple navigation instead of switch_page
                                        st.info(
                                            "Image viewer will be implemented in future phases"
                                        )

                                    # Delete button with confirmation
                                    if st.button(
                                        "🗑️ Delete",
                                        key=f"delete_{img_data['id']}",
                                        use_container_width=True,
                                    ):
                                        if st.session_state.get(
                                            f"confirm_delete_{img_data['id']}", False
                                        ):
                                            try:
                                                api_client.delete_image(img_data["id"])
                                                st.success(
                                                    f"Deleted {img_data['filename']}"
                                                )
                                                del st.session_state[
                                                    f"confirm_delete_{img_data['id']}"
                                                ]
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Delete failed: {str(e)}")
                                        else:
                                            st.session_state[
                                                f"confirm_delete_{img_data['id']}"
                                            ] = True
                                            st.warning(
                                                "⚠️ Click delete again to confirm"
                                            )

                                    # Enhanced metadata display
                                    if st.checkbox(
                                        "Show details", key=f"details_{img_data['id']}"
                                    ):
                                        st.json(
                                            {
                                                "ID": img_data["id"],
                                                "Filename": img_data["filename"],
                                                "Dimensions": f"{img_data['width']}x{img_data['height']}",
                                                "File Size": f"{file_size_mb:.2f} MB",
                                                "Format": img_data.get(
                                                    "format", "Unknown"
                                                ),
                                                "Upload Time": img_data.get(
                                                    "upload_time", "Unknown"
                                                ),
                                                "Annotations": img_data[
                                                    "annotation_count"
                                                ],
                                                "AI Predictions": img_data.get(
                                                    "has_ai_predictions", False
                                                ),
                                                "Processing Status": img_data.get(
                                                    "processing_status", "Unknown"
                                                ),
                                            }
                                        )

                                except Exception as e:
                                    st.error(
                                        f"❌ Error loading image {img_data.get('filename', 'Unknown')}: {str(e)}"
                                    )

                                st.markdown("---")

        except Exception as e:
            st.error(f"❌ **Error loading image gallery:** {str(e)}")

            with st.expander("🔧 Debug Information"):
                st.code(f"Error details: {str(e)}")

# === PAGE: ANNOTATION TOOL ===
elif page == "📝 Annotation Tool":
    st.header("📝 Annotation Tool")
    st.markdown("Draw bounding boxes and assign tags to UI elements in your images.")

    # Image selection
    st.subheader("🖼️ Select Image to Annotate")

    try:
        images = api_client.list_images()

        if not images:
            st.warning(
                "📥 **No images available**\n\nPlease upload some images first in the Image Management section."
            )
        else:
            # Image selection dropdown
            image_options = {
                f"{img['filename']} ({img['id'][:8]}...)": img["id"] for img in images
            }
            selected_display = st.selectbox(
                "Choose an image to annotate:",
                options=list(image_options.keys()),
                help="Select an image from your uploaded images",
            )

            if selected_display:
                selected_image_id = image_options[selected_display]
                selected_image = next(
                    img for img in images if img["id"] == selected_image_id
                )

                # Display image info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "📏 Dimensions",
                        f"{selected_image['width']}×{selected_image['height']}",
                    )
                with col2:
                    st.metric(
                        "📝 Annotations", selected_image.get("annotation_count", 0)
                    )
                with col3:
                    st.metric("📄 Format", selected_image["format"])

                st.markdown("---")

                # Load image data and existing annotations
                try:
                    image_content = api_client.get_image_file(selected_image_id)
                    existing_annotations = api_client.get_annotations(selected_image_id)

                    # Import annotation canvas components
                    import io

                    from components.annotation_canvas import (
                        annotation_controls,
                        create_annotation_canvas,
                    )

                    # Create annotation interface
                    st.subheader("🎨 Annotation Canvas")

                    # Create the annotation canvas
                    image_bytes = io.BytesIO(image_content)
                    session = create_annotation_canvas(
                        image_data=image_bytes,
                        image_id=selected_image_id,
                        existing_annotations=existing_annotations,
                        session_key=f"annotation_session_{selected_image_id}",
                    )

                    # Annotation controls
                    if session and hasattr(session, "draft_annotations"):
                        st.subheader("🛠️ Annotation Controls")
                        action = annotation_controls(session)
                    else:
                        action = None

                    # Handle save action
                    if action == "save":
                        try:
                            with st.spinner("💾 Saving annotations..."):
                                # Get annotations in API format
                                annotations_to_save = session.get_annotations_for_api()

                                # Save via API
                                result = api_client.save_annotation_batch(
                                    image_id=selected_image_id,
                                    created_by="streamlit_user",  # Could be made configurable
                                    annotations=annotations_to_save,
                                )

                                # Show results
                                if result.get("saved_count", 0) > 0:
                                    st.success(
                                        f"✅ Successfully saved {result['saved_count']} annotations!"
                                    )

                                    # Show conflicts if any
                                    if result.get("conflicts"):
                                        st.warning(
                                            f"⚠️ {len(result['conflicts'])} conflicts detected:"
                                        )
                                        for conflict in result["conflicts"]:
                                            st.write(
                                                f"- Overlap with existing annotation (IoU: {conflict.get('iou_score', 0):.2f})"
                                            )

                                    # Show warnings if any
                                    if result.get("warnings"):
                                        for warning in result["warnings"]:
                                            st.warning(f"⚠️ Warning: {warning}")

                                    # Clear the session after successful save
                                    session.draft_annotations = []

                                    # Show processing time
                                    st.info(
                                        f"⏱️ Processing time: {result.get('processing_time', 0):.2f} seconds"
                                    )

                                    # Trigger refresh
                                    st.rerun()
                                else:
                                    st.error("❌ No annotations were saved")

                        except Exception as e:
                            st.error(f"❌ Error saving annotations: {str(e)}")

                            # Show API error details if available
                            if hasattr(e, "response") and hasattr(e.response, "json"):
                                try:
                                    error_detail = e.response.json()
                                    if "errors" in error_detail.get("detail", {}):
                                        st.write("**Validation Errors:**")
                                        for error in error_detail["detail"]["errors"]:
                                            st.write(
                                                f"- {error['field']}: {error['message']}"
                                            )
                                except:
                                    pass

                    # Display existing annotations summary
                    if existing_annotations:
                        st.markdown("---")
                        st.subheader("📋 Existing Annotations")

                        # Summary by tag
                        tag_counts = {}
                        for ann in existing_annotations:
                            tag = ann.get("tag", "unknown")
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1

                        cols = st.columns(len(tag_counts) if tag_counts else 1)
                        for i, (tag, count) in enumerate(tag_counts.items()):
                            with cols[i % len(cols)]:
                                st.metric(f"{tag.title()}", count)

                        # Detailed list
                        with st.expander(
                            f"View all {len(existing_annotations)} annotations"
                        ):
                            for i, ann in enumerate(existing_annotations):
                                bbox = ann["bounding_box"]
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    # Show annotation info
                                    base_info = f"**{i+1}.** {ann['tag'].title()} at ({bbox['x']}, {bbox['y']}) {bbox['width']}×{bbox['height']}"
                                    st.write(f"{base_info}")
                                with col2:
                                    st.write(f"Status: {ann.get('status', 'active')}")

                except Exception as e:
                    st.error(f"❌ Error loading image or annotations: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error loading images: {str(e)}")
        st.code(f"Error details: {str(e)}")

elif page == "🤖 AI Predictions":
    st.header("🤖 AI Predictions")
    st.markdown(
        "Generate automatic UI element predictions using AI models with optional context awareness."
    )

    # Image selection
    st.subheader("🖼️ Select Image for Prediction")

    try:
        images = api_client.list_images()

        if not images:
            st.warning(
                "📥 **No images available**\n\nPlease upload some images first in the Image Management section."
            )
        else:
            # Image selection dropdown
            image_options = {
                f"{img['filename']} ({img['id'][:8]}...)": img["id"] for img in images
            }
            selected_display = st.selectbox(
                "Choose an image for AI prediction:",
                options=list(image_options.keys()),
                help="Select an image to generate UI element predictions",
            )

            if selected_display:
                selected_image_id = image_options[selected_display]
                selected_image = next(
                    img for img in images if img["id"] == selected_image_id
                )

                # Display image info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "📏 Dimensions",
                        f"{selected_image['width']}×{selected_image['height']}",
                    )
                with col2:
                    st.metric(
                        "📝 Annotations", selected_image.get("annotation_count", 0)
                    )
                with col3:
                    st.metric("📄 Format", selected_image["format"])
                with col4:
                    ai_status = (
                        "✅" if selected_image.get("has_ai_predictions") else "❌"
                    )
                    st.metric("🤖 AI Predictions", ai_status)

                st.markdown("---")

                # Prediction method selection
                st.subheader("🧠 Prediction Methods")

                prediction_col1, prediction_col2 = st.columns(2)

                with prediction_col1:
                    st.markdown("### 📡 Direct API Prediction")
                    st.info(
                        "**Traditional Method**\n\n"
                        "Uses OpenAI GPT-4V directly without context awareness. "
                        "Fast and reliable, but doesn't consider existing annotations."
                    )

                    if st.button(
                        "🚀 Generate Direct Predictions",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner(
                            "🤖 Generating predictions using direct API..."
                        ):
                            try:
                                predictions = api_client.generate_predictions(
                                    selected_image_id
                                )
                                st.success(
                                    f"✅ Generated {len(predictions.get('predictions', []))} predictions!"
                                )

                                # Display prediction results
                                if predictions.get("predictions"):
                                    st.subheader("📊 Prediction Results")

                                    # Summary metrics
                                    result_col1, result_col2, result_col3 = st.columns(
                                        3
                                    )
                                    with result_col1:
                                        st.metric(
                                            "Elements Found",
                                            len(predictions["predictions"]),
                                        )
                                    with result_col2:
                                        st.metric(
                                            "Processing Time",
                                            f"{predictions.get('processing_time', 0):.2f}s",
                                        )
                                    with result_col3:
                                        avg_conf = sum(
                                            p.get("confidence", 0)
                                            for p in predictions["predictions"]
                                        ) / len(predictions["predictions"])
                                        st.metric("Avg Confidence", f"{avg_conf:.2f}")

                                    # Detailed results
                                    with st.expander("View Detailed Results"):
                                        for i, pred in enumerate(
                                            predictions["predictions"]
                                        ):
                                            bbox = pred["bounding_box"]
                                            st.write(
                                                f"**{i+1}.** {pred['tag']} at ({bbox['x']}, {bbox['y']}) "
                                                f"{bbox['width']}×{bbox['height']} (confidence: {pred.get('confidence', 0):.2f})"
                                            )

                            except Exception as e:
                                st.error(f"❌ Prediction failed: {str(e)}")

                with prediction_col2:
                    st.markdown("### 🧠 MCP Context-Aware Prediction")

                    # Check if there are existing annotations for context
                    existing_count = selected_image.get("annotation_count", 0)
                    if existing_count > 0:
                        st.success(
                            f"**Enhanced Method** ✨\n\n"
                            f"Uses Model Context Protocol with awareness of {existing_count} existing annotations. "
                            f"Provides better accuracy by avoiding overlaps and finding missed elements."
                        )
                    else:
                        st.info(
                            "**Enhanced Method**\n\n"
                            "Uses Model Context Protocol for structured predictions. "
                            "Will provide better quality than direct API even without existing annotations."
                        )

                    if st.button(
                        "🧠 Generate MCP Predictions",
                        type="secondary",
                        use_container_width=True,
                    ):
                        with st.spinner(
                            "🤖 Generating context-aware predictions using MCP..."
                        ):
                            try:
                                mcp_predictions = api_client.generate_mcp_predictions(
                                    selected_image_id
                                )

                                if mcp_predictions.get("status") == "completed":
                                    st.success(
                                        f"✅ Generated {mcp_predictions.get('total_elements', 0)} context-aware predictions!"
                                    )

                                    # Display MCP prediction results
                                    st.subheader("📊 MCP Prediction Results")

                                    # Summary metrics
                                    (
                                        result_col1,
                                        result_col2,
                                        result_col3,
                                        result_col4,
                                    ) = st.columns(4)
                                    with result_col1:
                                        st.metric(
                                            "Elements Found",
                                            mcp_predictions.get("total_elements", 0),
                                        )
                                    with result_col2:
                                        st.metric(
                                            "Processing Time",
                                            f"{mcp_predictions.get('processing_time', 0):.2f}s",
                                        )
                                    with result_col3:
                                        st.metric(
                                            "Model Version",
                                            mcp_predictions.get(
                                                "model_version", "Unknown"
                                            ),
                                        )
                                    with result_col4:
                                        context_used = (
                                            "✅"
                                            if mcp_predictions.get(
                                                "context_used", False
                                            )
                                            else "❌"
                                        )
                                        st.metric("Context Used", context_used)

                                    # Detailed results
                                    if mcp_predictions.get("elements"):
                                        with st.expander("View Detailed MCP Results"):
                                            for i, elem in enumerate(
                                                mcp_predictions["elements"]
                                            ):
                                                bbox = elem["bounding_box"]
                                                reasoning = elem.get(
                                                    "reasoning", "No reasoning provided"
                                                )
                                                st.write(
                                                    f"**{i+1}.** {elem['tag']} at ({bbox['x']}, {bbox['y']}) "
                                                    f"{bbox['width']}×{bbox['height']} (confidence: {elem.get('confidence', 0):.2f})"
                                                )
                                                st.caption(f"💭 {reasoning}")

                                    # Show context information if used
                                    if mcp_predictions.get("context_used", False):
                                        st.info(
                                            f"🎯 This prediction considered {existing_count} existing annotations for better accuracy."
                                        )

                                else:
                                    st.warning(
                                        f"⚠️ MCP prediction completed with status: {mcp_predictions.get('status', 'unknown')}"
                                    )

                            except Exception as e:
                                st.error(f"❌ MCP prediction failed: {str(e)}")
                                st.info(
                                    "💡 MCP might not be available. Try the Direct API prediction instead."
                                )

                st.markdown("---")

                # Load and display existing predictions if available
                try:
                    existing_predictions = api_client.get_predictions(selected_image_id)
                    existing_annotations = api_client.get_annotations(selected_image_id)

                    if existing_predictions or existing_annotations:
                        st.subheader("📋 Current Data Summary")

                        summary_col1, summary_col2 = st.columns(2)

                        with summary_col1:
                            if existing_annotations:
                                st.markdown("**Manual Annotations:**")
                                tag_counts = {}
                                for ann in existing_annotations:
                                    tag = ann.get("tag", "unknown")
                                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

                                for tag, count in tag_counts.items():
                                    st.write(f"• {tag.title()}: {count}")
                            else:
                                st.write("No manual annotations yet")

                        with summary_col2:
                            if existing_predictions:
                                st.markdown("**AI Predictions:**")
                                pred_counts = {}
                                for pred in existing_predictions.get("predictions", []):
                                    tag = pred.get("tag", "unknown")
                                    pred_counts[tag] = pred_counts.get(tag, 0) + 1

                                for tag, count in pred_counts.items():
                                    st.write(f"• {tag.title()}: {count}")

                                st.caption(
                                    f"Model: {existing_predictions.get('llm_model', 'Unknown')}"
                                )
                            else:
                                st.write("No AI predictions yet")

                except Exception as e:
                    st.warning(f"Could not load existing data: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error loading images: {str(e)}")
        with st.expander("🔧 Debug Information"):
            st.code(f"Error details: {str(e)}")

elif page == "🎨 Enhanced Annotation Viewer":
    try:
        from pages.enhanced_annotation_viewer import show_enhanced_annotation_viewer

        show_enhanced_annotation_viewer()
    except ImportError as e:
        st.error(f"❌ Could not load Enhanced Annotation Viewer: {str(e)}")
        st.info("Please ensure all required dependencies are installed.")
        with st.expander("🔧 Debug Information"):
            st.code(f"Import error details: {str(e)}")
            st.code(
                "Current working directory and Python path info will help debug this issue."
            )

elif page == "📊 Analytics & Export":
    st.header("📊 Analytics & Export")
    st.info(
        "🚧 **Phase 1.1 Implementation**\n\nAnalytics and export features will be implemented in future phases."
    )

# Footer information
st.markdown("---")
st.markdown(
    "**UI Component Labeling System v1.0** | Phase 1.1 Implementation | Enhanced Upload Flow with Comprehensive Validation"
)
