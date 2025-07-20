import io
import json
from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw

import streamlit as st
from utils.api_client import UILabelingAPIClient

# Page configuration
st.set_page_config(
    page_title="UI Component Labeling System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize API client
@st.cache_resource
def get_api_client():
    return UILabelingAPIClient()


api_client = get_api_client()

# Custom CSS for better UI
st.markdown(
    """
<style>
    .annotation-box {
        border: 2px solid #ff4b4b;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f2f6;
    }
    .prediction-box {
        border: 2px solid #1f77b4;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #e8f4f8;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title("🎨 UI Component Labeling System")
st.markdown(
    "Upload UI screenshots, draw bounding boxes, and evaluate AI-powered component detection"
)

# Check API connection
if not api_client.is_api_available():
    st.error("⚠️ Cannot connect to the API. Please make sure the backend is running.")
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Choose a page:",
        [
            "🏠 Image Management",
            "✏️ Annotation Tool",
            "🤖 AI Predictions",
            "📊 Evaluation & Statistics",
            "🔧 System Status",
        ],
    )

# === PAGE: IMAGE MANAGEMENT ===
if page == "🏠 Image Management":
    st.header("Image Management")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Upload New Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            help="Upload a UI screenshot or design image for labeling",
        )

        if uploaded_file is not None:
            if st.button("Upload Image"):
                try:
                    with st.spinner("Uploading image..."):
                        file_content = uploaded_file.read()
                        response = api_client.upload_image(
                            file_content, uploaded_file.name
                        )
                        st.success(f"Image uploaded successfully! ID: {response['id']}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error uploading image: {str(e)}")

    with col2:
        st.subheader("Image Library")

        try:
            images = api_client.list_images()

            if not images:
                st.info(
                    "No images uploaded yet. Upload your first image to get started!"
                )
            else:
                # Create image grid
                cols_per_row = 3
                for i in range(0, len(images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(images):
                            img_data = images[i + j]
                            with col:
                                try:
                                    # Get image thumbnail
                                    img_content = api_client.get_image_file(
                                        img_data["id"]
                                    )
                                    image = Image.open(io.BytesIO(img_content))

                                    # Display thumbnail
                                    st.image(
                                        image,
                                        caption=img_data["filename"],
                                        use_column_width=True,
                                    )

                                    # Image info
                                    st.write(
                                        f"**Size:** {img_data['width']}x{img_data['height']}"
                                    )
                                    st.write(
                                        f"**Annotations:** {img_data['annotation_count']}"
                                    )
                                    st.write(
                                        f"**AI Predictions:** {'✅' if img_data['has_ai_predictions'] else '❌'}"
                                    )

                                    # Action buttons
                                    col_edit, col_delete = st.columns(2)
                                    with col_edit:
                                        if st.button(
                                            f"Edit", key=f"edit_{img_data['id']}"
                                        ):
                                            st.session_state.selected_image_id = (
                                                img_data["id"]
                                            )
                                            st.session_state.page = "✏️ Annotation Tool"
                                            st.rerun()
                                    with col_delete:
                                        if st.button(
                                            f"Delete", key=f"delete_{img_data['id']}"
                                        ):
                                            try:
                                                api_client.delete_image(img_data["id"])
                                                st.success(
                                                    "Image deleted successfully!"
                                                )
                                                st.rerun()
                                            except Exception as e:
                                                st.error(
                                                    f"Error deleting image: {str(e)}"
                                                )

                                except Exception as e:
                                    st.error(f"Error loading image: {str(e)}")

        except Exception as e:
            st.error(f"Error loading images: {str(e)}")

# === PAGE: ANNOTATION TOOL ===
elif page == "✏️ Annotation Tool":
    st.header("Interactive Annotation Tool")

    # Image selection
    try:
        images = api_client.list_images()
        if not images:
            st.warning("No images available. Please upload an image first.")
            st.stop()

        # Get selected image ID from session state or selectbox
        selected_image_id = st.session_state.get("selected_image_id")

        image_options = {img["filename"]: img["id"] for img in images}
        selected_filename = st.selectbox(
            "Select an image to annotate:",
            options=list(image_options.keys()),
            index=(
                list(image_options.values()).index(selected_image_id)
                if selected_image_id in image_options.values()
                else 0
            ),
        )
        selected_image_id = image_options[selected_filename]
        st.session_state.selected_image_id = selected_image_id

    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        st.stop()

    # Load image and annotations
    try:
        img_content = api_client.get_image_file(selected_image_id)
        image = Image.open(io.BytesIO(img_content))
        annotations = api_client.get_annotations(selected_image_id)

        # Import our custom components
        from components.annotation_canvas import (
            AnnotationCanvas,
            create_annotation_summary,
            validate_annotation_bounds,
        )
        from components.image_viewer import ImageViewer

        # Create tabs for different annotation methods
        tab1, tab2, tab3 = st.tabs(
            ["🎨 Interactive Drawing", "📝 Manual Entry", "📊 Overview"]
        )

        with tab1:
            st.subheader("Draw Bounding Boxes")
            st.markdown(
                "**Instructions:** Select a component type, then draw rectangles on the image by clicking and dragging."
            )

            # Create annotation canvas
            canvas = AnnotationCanvas(image, annotations)
            canvas_result = canvas.render_canvas(
                canvas_key=f"canvas_{selected_image_id}"
            )

            # Handle new annotations from canvas
            if canvas_result["new_annotations"]:
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(
                        f"**Found {len(canvas_result['new_annotations'])} new annotations:**"
                    )
                    for i, ann in enumerate(canvas_result["new_annotations"]):
                        bbox = ann["bounding_box"]
                        st.write(
                            f"- {ann['tag'].title()}: ({bbox['x']}, {bbox['y']}) - {bbox['width']}×{bbox['height']}"
                        )

                with col2:
                    annotator_name = st.text_input(
                        "Annotator name:", value="user", key="canvas_annotator"
                    )

                    if st.button("💾 Save All Annotations", type="primary"):
                        saved_count = 0
                        for ann in canvas_result["new_annotations"]:
                            # Validate annotation bounds
                            if validate_annotation_bounds(
                                ann, image.width, image.height
                            ):
                                try:
                                    annotation_data = {
                                        "bounding_box": ann["bounding_box"],
                                        "tag": ann["tag"],
                                        "annotator": annotator_name,
                                    }
                                    api_client.create_annotation(
                                        selected_image_id, annotation_data
                                    )
                                    saved_count += 1
                                except Exception as e:
                                    st.error(f"Error saving annotation: {str(e)}")
                            else:
                                st.warning(
                                    f"Skipped invalid annotation: {ann['tag']} at ({ann['bounding_box']['x']}, {ann['bounding_box']['y']})"
                                )

                        if saved_count > 0:
                            st.success(f"Saved {saved_count} annotations!")
                            st.rerun()

        with tab2:
            st.subheader("Manual Annotation Entry")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Display image for reference
                viewer = ImageViewer(image)
                fig = viewer.render_with_annotations(
                    annotations=annotations, width=400, height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Manual entry form
                with st.form("manual_annotation"):
                    st.write("**Add New Annotation Manually**")
                    x = st.number_input(
                        "X coordinate", min_value=0, max_value=image.width, value=0
                    )
                    y = st.number_input(
                        "Y coordinate", min_value=0, max_value=image.height, value=0
                    )
                    width = st.number_input(
                        "Width", min_value=1, max_value=image.width, value=50
                    )
                    height = st.number_input(
                        "Height", min_value=1, max_value=image.height, value=30
                    )
                    tag = st.selectbox(
                        "Component Type", ["button", "input", "radio", "dropdown"]
                    )
                    annotator = st.text_input("Annotator Name", value="user")

                    if st.form_submit_button("Add Annotation"):
                        try:
                            annotation_data = {
                                "bounding_box": {
                                    "x": x,
                                    "y": y,
                                    "width": width,
                                    "height": height,
                                },
                                "tag": tag,
                                "annotator": annotator,
                            }

                            # Validate bounds
                            if validate_annotation_bounds(
                                annotation_data, image.width, image.height
                            ):
                                api_client.create_annotation(
                                    selected_image_id, annotation_data
                                )
                                st.success("Annotation added successfully!")
                                st.rerun()
                            else:
                                st.error(
                                    "Invalid annotation bounds. Please check coordinates and size."
                                )
                        except Exception as e:
                            st.error(f"Error adding annotation: {str(e)}")

        with tab3:
            st.subheader("Annotation Overview")

            if annotations:
                # Statistics
                summary = create_annotation_summary(annotations)
                total_annotations = sum(summary.values())

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Total", total_annotations)
                with col2:
                    st.metric("Buttons", summary["button"])
                with col3:
                    st.metric("Inputs", summary["input"])
                with col4:
                    st.metric("Radios", summary["radio"])
                with col5:
                    st.metric("Dropdowns", summary["dropdown"])

                # Annotation list with management
                canvas.render_annotation_list(annotations)

                # Handle annotation updates from the list
                for i, ann in enumerate(annotations):
                    if st.session_state.get(f"editing_{ann['id']}", False):
                        # This would be handled by the component's edit functionality
                        pass

                # Bulk operations
                st.subheader("Bulk Operations")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🗑️ Delete All Annotations", type="secondary"):
                        if st.session_state.get("confirm_delete_all"):
                            try:
                                for ann in annotations:
                                    api_client.delete_annotation(ann["id"])
                                st.success("All annotations deleted!")
                                st.session_state.confirm_delete_all = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting annotations: {str(e)}")
                        else:
                            st.session_state.confirm_delete_all = True
                            st.warning(
                                "Click again to confirm deletion of all annotations"
                            )

                with col2:
                    if st.button("📥 Export Annotations"):
                        from datetime import datetime

                        # Create downloadable JSON
                        export_data = {
                            "image_id": selected_image_id,
                            "image_filename": selected_filename,
                            "image_dimensions": {
                                "width": image.width,
                                "height": image.height,
                            },
                            "annotations": annotations,
                            "export_timestamp": datetime.now().isoformat(),
                        }

                        st.download_button(
                            label="💾 Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"annotations_{selected_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )

                with col3:
                    if st.button("🔄 Refresh"):
                        st.rerun()

            else:
                st.info(
                    "No annotations yet. Use the drawing tool or manual entry to create annotations."
                )

                # Show image without annotations
                viewer = ImageViewer(image)
                fig = viewer.render_with_annotations(width=600, height=400)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading annotation data: {str(e)}")

# === PAGE: AI PREDICTIONS ===
elif page == "🤖 AI Predictions":
    st.header("AI Predictions")

    try:
        images = api_client.list_images()
        if not images:
            st.warning("No images available. Please upload an image first.")
            st.stop()

        # Image selection
        image_options = {img["filename"]: img["id"] for img in images}
        selected_filename = st.selectbox(
            "Select an image for AI analysis:", options=list(image_options.keys())
        )
        selected_image_id = image_options[selected_filename]

        col1, col2 = st.columns([1, 1])

        with col1:
            # Generate predictions button
            if st.button("🤖 Generate AI Predictions", type="primary"):
                try:
                    with st.spinner(
                        "Generating AI predictions... This may take a moment."
                    ):
                        predictions = api_client.generate_predictions(selected_image_id)
                        st.success(
                            f"Generated {len(predictions['predictions'])} predictions!"
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")

        with col2:
            # Load existing predictions
            predictions = api_client.get_predictions(selected_image_id)
            if predictions:
                st.info(
                    f"✅ AI predictions available ({len(predictions['predictions'])} found)"
                )
                st.write(f"**Model:** {predictions['llm_model']}")
                st.write(f"**Processing Time:** {predictions['processing_time']:.2f}s")
            else:
                st.warning("❌ No AI predictions available for this image")

        # Display predictions if available
        if predictions:
            st.subheader("AI Predictions vs Ground Truth")

            # Load image and annotations
            img_content = api_client.get_image_file(selected_image_id)
            image = Image.open(io.BytesIO(img_content))
            annotations = api_client.get_annotations(selected_image_id)

            # Use the ImageViewer component for better visualization
            from components.image_viewer import ImageViewer

            viewer = ImageViewer(image)
            viewer.render_comparison_view(
                annotations=annotations, predictions=predictions["predictions"]
            )

            # Show prediction details
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ground Truth Annotations")
                for i, ann in enumerate(annotations):
                    st.markdown(
                        f"""
                    <div class="annotation-box">
                        <strong>Annotation {i+1}</strong><br>
                        Type: {ann['tag']}<br>
                        Position: ({ann['bounding_box']['x']}, {ann['bounding_box']['y']})<br>
                        Size: {ann['bounding_box']['width']} x {ann['bounding_box']['height']}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with col2:
                st.subheader("AI Predictions")
                for i, pred in enumerate(predictions["predictions"]):
                    confidence = pred.get("confidence", 0.0)
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <strong>Prediction {i+1}</strong><br>
                        Type: {pred['tag']}<br>
                        Confidence: {confidence:.2f}<br>
                        Position: ({pred['bounding_box']['x']}, {pred['bounding_box']['y']})<br>
                        Size: {pred['bounding_box']['width']} x {pred['bounding_box']['height']}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(f"Error loading prediction data: {str(e)}")

# === PAGE: EVALUATION & STATISTICS ===
elif page == "📊 Evaluation & Statistics":
    st.header("Evaluation & Statistics")

    try:
        # System-wide statistics
        stats = api_client.get_statistics()

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Images", stats["total_images"])
        with col2:
            st.metric("Total Annotations", stats["total_annotations"])
        with col3:
            st.metric("Images with AI Predictions", stats["images_with_predictions"])
        with col4:
            st.metric("Storage Used", f"{stats['total_storage_mb']} MB")

        # Per-image evaluation
        st.subheader("Per-Image Evaluation")

        images = api_client.list_images()
        images_with_both = [
            img
            for img in images
            if img["annotation_count"] > 0 and img["has_ai_predictions"]
        ]

        if not images_with_both:
            st.warning(
                "No images have both annotations and AI predictions for evaluation."
            )
        else:
            # Evaluation results
            evaluation_results = []

            for img in images_with_both:
                try:
                    eval_result = api_client.evaluate_predictions(img["id"])
                    eval_result["filename"] = img["filename"]
                    evaluation_results.append(eval_result)
                except Exception as e:
                    st.error(f"Error evaluating {img['filename']}: {str(e)}")

            if evaluation_results:
                # Create evaluation table
                import pandas as pd

                df = pd.DataFrame(evaluation_results)

                # Display table
                st.dataframe(
                    df[
                        [
                            "filename",
                            "total_annotations",
                            "total_predictions",
                            "matches",
                            "precision",
                            "recall",
                            "f1_score",
                        ]
                    ],
                    use_container_width=True,
                )

                # Create visualizations
                col1, col2 = st.columns(2)

                with col1:
                    # F1 Score chart
                    fig_f1 = px.bar(
                        df,
                        x="filename",
                        y="f1_score",
                        title="F1 Score by Image",
                        labels={"f1_score": "F1 Score", "filename": "Image"},
                    )
                    fig_f1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_f1, use_container_width=True)

                with col2:
                    # Precision vs Recall scatter
                    fig_pr = px.scatter(
                        df,
                        x="precision",
                        y="recall",
                        size="matches",
                        hover_data=["filename"],
                        title="Precision vs Recall",
                        labels={"precision": "Precision", "recall": "Recall"},
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)

                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    avg_precision = df["precision"].mean()
                    st.metric("Average Precision", f"{avg_precision:.3f}")

                with col2:
                    avg_recall = df["recall"].mean()
                    st.metric("Average Recall", f"{avg_recall:.3f}")

                with col3:
                    avg_f1 = df["f1_score"].mean()
                    st.metric("Average F1 Score", f"{avg_f1:.3f}")

                # Tag-wise analysis
                st.subheader("Component Type Analysis")

                # This would require more detailed analysis from the backend
                # For now, show basic statistics
                all_annotations = []
                all_predictions = []

                for img in images_with_both:
                    try:
                        annotations = api_client.get_annotations(img["id"])
                        predictions_data = api_client.get_predictions(img["id"])

                        if predictions_data:
                            all_annotations.extend(annotations)
                            all_predictions.extend(predictions_data["predictions"])
                    except:
                        continue

                if all_annotations and all_predictions:
                    # Count by tag type
                    ann_counts = {}
                    pred_counts = {}

                    for ann in all_annotations:
                        tag = ann["tag"]
                        ann_counts[tag] = ann_counts.get(tag, 0) + 1

                    for pred in all_predictions:
                        tag = pred["tag"]
                        pred_counts[tag] = pred_counts.get(tag, 0) + 1

                    # Create comparison chart
                    tags = list(set(list(ann_counts.keys()) + list(pred_counts.keys())))
                    ann_values = [ann_counts.get(tag, 0) for tag in tags]
                    pred_values = [pred_counts.get(tag, 0) for tag in tags]

                    fig_tags = go.Figure(
                        data=[
                            go.Bar(name="Ground Truth", x=tags, y=ann_values),
                            go.Bar(name="AI Predictions", x=tags, y=pred_values),
                        ]
                    )
                    fig_tags.update_layout(
                        barmode="group",
                        title="Component Count by Type",
                        xaxis_title="Component Type",
                        yaxis_title="Count",
                    )
                    st.plotly_chart(fig_tags, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

# === PAGE: SYSTEM STATUS ===
elif page == "🔧 System Status":
    st.header("System Status")

    try:
        health = api_client.get_health()

        # API Status
        if health["status"] == "healthy":
            st.success("✅ API is healthy and responsive")
        else:
            st.error("❌ API is experiencing issues")

        # LLM Service Status
        if health["llm_service_available"]:
            st.success("✅ LLM service is available")
        else:
            st.warning("⚠️ LLM service is not available (check OPENAI_API_KEY)")

        # Storage Statistics
        st.subheader("Storage Information")
        storage_stats = health["storage_stats"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images", storage_stats["total_images"])
        with col2:
            st.metric("Annotations", storage_stats["total_annotations"])
        with col3:
            st.metric("Predictions", storage_stats["total_predictions"])
        with col4:
            st.metric("Storage", f"{storage_stats['total_storage_mb']} MB")

        # API Information
        st.subheader("API Information")
        st.code(f"Base URL: {api_client.base_url}")

        # Test API endpoints
        st.subheader("API Endpoint Tests")

        endpoints_to_test = [
            ("Health Check", "/health"),
            ("List Images", "/images"),
            ("System Statistics", "/statistics"),
        ]

        for name, endpoint in endpoints_to_test:
            try:
                if endpoint == "/health":
                    api_client.get_health()
                elif endpoint == "/images":
                    api_client.list_images()
                elif endpoint == "/statistics":
                    api_client.get_statistics()

                st.success(f"✅ {name}: OK")
            except Exception as e:
                st.error(f"❌ {name}: {str(e)}")

    except Exception as e:
        st.error(f"Error checking system status: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*UI Component Labeling System - Phase 1 Implementation*")
