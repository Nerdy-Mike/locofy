import base64
import time
from typing import List, Optional, Tuple

import openai
from openai import OpenAI
from PIL import Image

from models.annotation_models import (
    Annotation,
    BoundingBox,
    LLMPrediction,
    UIElementTag,
)


class LLMUIDetectionService:
    """Service for automatic UI component detection using OpenAI GPT-4V"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.llm_model = "gpt-4o"  # Updated to current model

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get original image dimensions"""
        with Image.open(image_path) as img:
            return img.size

    def _preprocess_image_for_llm(self, image_path: str) -> Tuple[str, float, int, int]:
        """
        Preprocess image for LLM processing, handling resizing if needed

        Returns:
            Tuple of (processed_image_path, scale_factor, resized_width, resized_height)
            scale_factor = 1.0 if no resizing was done
            scale_factor < 1.0 if image was scaled down
        """
        try:
            # Get original dimensions
            original_width, original_height = self._get_image_dimensions(image_path)

            # Check if resizing is needed (same logic as UIImageValidationService)
            max_dimension = 1024

            if max(original_width, original_height) <= max_dimension:
                # No resizing needed
                return image_path, 1.0, original_width, original_height

            # Calculate resize ratio
            if original_width > original_height:
                scale_factor = max_dimension / original_width
                new_width = max_dimension
                new_height = int(original_height * scale_factor)
            else:
                scale_factor = max_dimension / original_height
                new_height = max_dimension
                new_width = int(original_width * scale_factor)

            # Resize image
            with Image.open(image_path) as img:
                resized_img = img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

                # Save to temporary file
                resized_path = f"{image_path}_llm_resized.jpg"
                resized_img.convert("RGB").save(resized_path, "JPEG", quality=85)

                print(
                    f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} "
                    f"(scale factor: {scale_factor:.3f}) for LLM processing"
                )

                return resized_path, scale_factor, new_width, new_height

        except Exception as e:
            print(f"Failed to preprocess image {image_path}: {e}")
            return (
                image_path,
                1.0,
                original_width,
                original_height,
            )  # Return original on error

    def _scale_coordinates_to_original(
        self, bbox_data: dict, scale_factor: float
    ) -> dict:
        """Scale coordinates from resized image back to original image dimensions"""
        if scale_factor == 1.0:
            return bbox_data

        return {
            "x": bbox_data["x"] / scale_factor,
            "y": bbox_data["y"] / scale_factor,
            "width": bbox_data["width"] / scale_factor,
            "height": bbox_data["height"] / scale_factor,
        }

    def _cleanup_temp_file(self, file_path: str, original_path: str):
        """Clean up temporary file if it was created"""
        if file_path != original_path:
            try:
                import os

                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"Failed to cleanup temporary file {file_path}: {e}")

    def _validate_and_scale_coordinates(
        self,
        bbox_data: dict,
        scale_factor: float,
        resized_width: int,
        resized_height: int,
    ) -> dict:
        """Validate coordinates are within bounds before scaling back to original dimensions"""

        # First validate that coordinates are within the resized image bounds
        x, y, width, height = (
            bbox_data["x"],
            bbox_data["y"],
            bbox_data["width"],
            bbox_data["height"],
        )

        # Check bounds on resized image
        if (
            x < 0
            or y < 0
            or x + width > resized_width
            or y + height > resized_height
            or width <= 0
            or height <= 0
        ):

            print(
                f"WARNING: LLM provided out-of-bounds coordinates: "
                f"({x}, {y}) {width}×{height} on {resized_width}×{resized_height} image. "
                f"Clamping to valid range."
            )

            # Clamp coordinates to valid range
            x = max(0, min(x, resized_width - 1))
            y = max(0, min(y, resized_height - 1))
            width = max(1, min(width, resized_width - x))
            height = max(1, min(height, resized_height - y))

            bbox_data = {"x": x, "y": y, "width": width, "height": height}

        # Now scale to original dimensions
        if scale_factor == 1.0:
            return bbox_data

        return {
            "x": bbox_data["x"] / scale_factor,
            "y": bbox_data["y"] / scale_factor,
            "width": bbox_data["width"] / scale_factor,
            "height": bbox_data["height"] / scale_factor,
        }

    def detect_ui_components(self, image_id: str, image_path: str) -> LLMPrediction:
        """Detect UI components in an image using GPT-4V"""
        start_time = time.time()
        processed_image_path = image_path

        try:
            # Get original image dimensions for prompt context
            original_width, original_height = self._get_image_dimensions(image_path)

            # Preprocess image (resize if needed) and get scale factor
            processed_image_path, scale_factor, resized_width, resized_height = (
                self._preprocess_image_for_llm(image_path)
            )

            # Encode the processed image
            base64_image = self.encode_image(processed_image_path)

            # Create the prompt for UI component detection with improved accuracy
            prompt = f"""
            You are an expert UI/UX analyst. Analyze this UI screenshot and identify all interactive UI components with high precision.
            
            DETECTION GUIDELINES:
            1. Be generous with button boundaries - include the entire clickable area plus any visual styling
            2. Look for ALL interactive elements, including modern styled buttons
            3. Include button text, borders, padding, background, and hover areas in the bounding box
            4. Pay attention to visual hierarchy - larger elements are often more important
            5. Consider modern UI patterns: flat design, subtle shadows, color-based buttons
            
            MODERN UI BUTTON PATTERNS TO DETECT:
            - Primary buttons (often brightly colored, high contrast)
            - Secondary buttons (subtle styling, outlined or muted colors)
            - Text buttons (clickable text with subtle hover states)
            - Icon buttons (icons with clickable areas)
            - Navigation elements (nav items, tabs, breadcrumbs)
            - Call-to-action buttons (prominent styling, "Get Started", "Sign Up", etc.)
            
            For each UI component you find, provide:
            1. Precise bounding box coordinates (x, y, width, height) where (0,0) is top-left
            2. Component type: button, input, radio, or dropdown
            3. A confidence score between 0.0 and 1.0
            4. Brief description of what you detected
            
            BUTTON DETECTION TIPS:
            - Include the full visual area (text + background + borders + padding + margin)
            - Look for rectangular clickable areas with text or icons
            - Consider hover states and visual affordances (subtle shadows, color changes)
            - Don't be too conservative with boundaries - err on the side of larger, more complete boxes
            - Main action buttons are often larger and more prominent
            - Secondary buttons may have subtle styling but clear clickable areas
            - Look for text that appears clickable (links, nav items, action text)
            
            THEME-SPECIFIC GUIDANCE:
            - Dark themes: Look for subtle color variations, lighter text on dark backgrounds
            - Light themes: Look for darker elements, shadows, borders
            - Modern flat design: Buttons may only differ by background color or subtle borders
            - Component libraries: Often have consistent button styling patterns
            
            🖼️ CRITICAL COORDINATE INFORMATION:
            - Image dimensions: {resized_width} × {resized_height} pixels
            - Coordinate system: (0,0) is TOP-LEFT corner
            - Maximum valid coordinates: x < {resized_width}, y < {resized_height}
            - ALL coordinates must be within image bounds!
            
            ⚠️ COORDINATE VALIDATION RULES:
            - x coordinate: 0 ≤ x < {resized_width}
            - y coordinate: 0 ≤ y < {resized_height}  
            - width: 1 ≤ width ≤ {resized_width}
            - height: 1 ≤ height ≤ {resized_height}
            - x + width ≤ {resized_width}
            - y + height ≤ {resized_height}
            
            COORDINATE ACCURACY:
            - This image is {resized_width}×{resized_height} pixels
            - Use precise pixel coordinates within the image bounds
            - Ensure bounding boxes fully contain the UI elements including any visual styling
            - Include adequate padding around text to cover full clickable area
            - Double-check all coordinates are within the valid range!
            
            Return your response as a JSON array with this exact format:
            [
              {{
                "type": "button|input|radio|dropdown",
                "x": number,
                "y": number, 
                "width": number,
                "height": number,
                "confidence": number,
                "description": "brief description of what you detected (e.g., 'primary CTA button', 'navigation link', 'secondary action button')"
              }}
            ]
            
            Focus on clearly visible interactive elements. Be thorough and include all clickable areas.
            Look for modern button patterns including subtle styling and text-based buttons.
            ENSURE ALL COORDINATES ARE WITHIN THE IMAGE BOUNDS: {resized_width}×{resized_height}!
            Only return the JSON array, no additional text.
            """

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.1,
            )

            # Parse the response
            content = response.choices[0].message.content.strip()

            # Clean up the response to extract JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            # Parse JSON response
            import json

            detections = json.loads(content)

            # Convert to Annotation objects with coordinate scaling
            annotations = []
            for detection in detections:
                try:
                    # Map type to UIElementTag
                    tag_mapping = {
                        "button": UIElementTag.BUTTON,
                        "input": UIElementTag.INPUT,
                        "radio": UIElementTag.RADIO,
                        "dropdown": UIElementTag.DROPDOWN,
                    }

                    ui_tag = tag_mapping.get(detection["type"].lower())
                    if not ui_tag:
                        continue

                    # Scale coordinates back to original image dimensions
                    scaled_coords = self._validate_and_scale_coordinates(
                        {
                            "x": detection["x"],
                            "y": detection["y"],
                            "width": detection["width"],
                            "height": detection["height"],
                        },
                        scale_factor,
                        resized_width,  # Pass resized width
                        resized_height,  # Pass resized height
                    )

                    # Create bounding box with scaled coordinates
                    bbox = BoundingBox(
                        x=float(scaled_coords["x"]),
                        y=float(scaled_coords["y"]),
                        width=float(scaled_coords["width"]),
                        height=float(scaled_coords["height"]),
                    )

                    # Create annotation
                    annotation = Annotation(
                        image_id=image_id,
                        bounding_box=bbox,
                        tag=ui_tag,
                        confidence=float(detection["confidence"]),
                        annotator="gpt-4v",
                    )

                    annotations.append(annotation)

                except (KeyError, ValueError, TypeError) as e:
                    print(f"Error parsing detection: {detection}, error: {e}")
                    continue

            # Log coordinate scaling info if scaling was applied
            if scale_factor != 1.0:
                print(
                    f"Scaled {len(annotations)} annotations from resized image (scale factor: {scale_factor:.3f}) back to original dimensions"
                )

            processing_time = time.time() - start_time

            return LLMPrediction(
                image_id=image_id,
                predictions=annotations,
                llm_model=self.llm_model,
                processing_time=processing_time,
            )

        except Exception as e:
            print(f"Error in LLM detection: {e}")
            # Return empty prediction on error
            return LLMPrediction(
                image_id=image_id,
                predictions=[],
                llm_model=self.llm_model,
                processing_time=time.time() - start_time,
            )

        finally:
            # Clean up temporary resized file if it was created
            self._cleanup_temp_file(processed_image_path, image_path)

    def detect_with_custom_prompt(
        self, image_id: str, image_path: str, custom_prompt: str
    ) -> LLMPrediction:
        """Detect UI components with a custom prompt"""
        start_time = time.time()

        # Encode image
        base64_image = self.encode_image(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": custom_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.1,
            )

            # For custom prompts, we might get different response formats
            # This is a simplified version - you might want to add custom parsing logic
            content = response.choices[0].message.content.strip()

            processing_time = time.time() - start_time

            return LLMPrediction(
                image_id=image_id,
                predictions=[],  # Custom prompt results would need custom parsing
                llm_model=self.llm_model,
                processing_time=processing_time,
            )

        except Exception as e:
            print(f"Error in custom LLM detection: {e}")
            return LLMPrediction(
                image_id=image_id,
                predictions=[],
                llm_model=self.llm_model,
                processing_time=time.time() - start_time,
            )
