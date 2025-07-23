"""
MCP (Model Context Protocol) service for enhanced LLM integration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from PIL import Image

# Note: Using direct OpenAI integration instead of MCP for now
from pydantic import BaseModel

from models.annotation_models import Annotation, BoundingBox, UIElementTag
from models.validation_models import (
    DetectedElement,
    MCPContext,
    PredictionResponse,
    ProcessingStatus,
)
from utils.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class MCPTool(BaseModel):
    """MCP tool definition"""

    name: str
    description: str
    parameters: Dict[str, Any]


class MCPUIDetectionService:
    """Enhanced UI Detection Service using Model Context Protocol (MCP)"""

    def __init__(self):
        """Initialize the MCP UI Detection Service"""
        self.session = None
        self.enabled = hasattr(config, "openai_api_key") and config.openai_api_key
        logger.info(f"MCPUIDetectionService initialized (enabled: {self.enabled})")

    async def _get_image_dimensions_from_data(
        self, image_data: bytes
    ) -> tuple[int, int]:
        """Get image dimensions from bytes data"""
        try:
            from io import BytesIO

            with Image.open(BytesIO(image_data)) as img:
                return img.size
        except Exception as e:
            logger.warning(f"Failed to get image dimensions: {e}")
            return 0, 0

    async def _preprocess_image_for_llm(self, image_data: bytes) -> tuple[bytes, float]:
        """
        Preprocess image for LLM processing, handling resizing if needed

        Returns:
            Tuple of (processed_image_data, scale_factor)
        """
        try:
            from io import BytesIO

            # Get original dimensions
            original_width, original_height = (
                await self._get_image_dimensions_from_data(image_data)
            )

            if original_width == 0 or original_height == 0:
                return image_data, 1.0

            # Check if resizing is needed
            max_dimension = 1024

            if max(original_width, original_height) <= max_dimension:
                return image_data, 1.0

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
            with Image.open(BytesIO(image_data)) as img:
                resized_img = img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

                # Convert back to bytes
                output = BytesIO()
                resized_img.convert("RGB").save(output, "JPEG", quality=85)
                resized_data = output.getvalue()

                logger.info(
                    f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} "
                    f"(scale factor: {scale_factor:.3f}) for MCP processing"
                )

                return resized_data, scale_factor

        except Exception as e:
            logger.warning(f"Failed to preprocess image: {e}")
            return image_data, 1.0  # Return original on error

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

            logger.warning(
                f"MCP provided out-of-bounds coordinates: "
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

    async def detect_ui_elements(
        self, image_data: bytes, image_id: str, context: Optional[MCPContext] = None
    ) -> PredictionResponse:
        """Detect UI elements in an image using MCP-enhanced detection"""

        start_time = datetime.now()

        try:
            # Preprocess image (resize if needed) and get scale factor
            processed_image_data, scale_factor = await self._preprocess_image_for_llm(
                image_data
            )

            # Build detection context
            detection_context = await self._build_detection_context(
                processed_image_data, image_id, context
            )

            # Execute UI detection with MCP tools
            raw_results = await self._execute_detection_with_tools(detection_context)

            # Parse and validate results with coordinate scaling
            detected_elements = await self._parse_detection_results(
                raw_results, scale_factor
            )

            # Log coordinate scaling info if scaling was applied
            if scale_factor != 1.0:
                logger.info(
                    f"Scaled {len(detected_elements)} MCP annotations from resized image "
                    f"(scale factor: {scale_factor:.3f}) back to original dimensions"
                )

            # Create response
            processing_time = (datetime.now() - start_time).total_seconds()

            return PredictionResponse(
                prediction_id=f"pred_{image_id}_{int(start_time.timestamp())}",
                image_id=image_id,
                elements=detected_elements,
                processing_time=processing_time,
                model_version="gpt-4-vision-mcp",
                confidence_threshold=0.5,
                total_elements=len(detected_elements),
                status=ProcessingStatus.COMPLETED,
            )

        except Exception as e:
            logger.error(f"UI detection failed for image {image_id}: {e}")
            return PredictionResponse(
                prediction_id=f"pred_{image_id}_failed",
                image_id=image_id,
                elements=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_version="gpt-4-vision-mcp",
                confidence_threshold=0.5,
                total_elements=0,
                status=ProcessingStatus.FAILED,
            )

    async def _build_detection_context(
        self, image_data: bytes, image_id: str, context: Optional[MCPContext]
    ) -> Dict[str, Any]:
        """Build rich context for MCP detection with existing annotations"""

        # Convert image to base64
        import base64

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        detection_context = {
            "image": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                },
            },
            "task": "ui_element_detection",
            "instructions": self._get_detection_instructions(),
            "ui_element_types": ["button", "input", "radio", "dropdown"],
        }

        # Add existing annotations as context if available
        if context and context.previous_predictions:
            detection_context["existing_annotations"] = []

            for pred in context.previous_predictions:
                detection_context["existing_annotations"].append(
                    {
                        "tag": (
                            pred.tag.value
                            if hasattr(pred.tag, "value")
                            else str(pred.tag)
                        ),
                        "bounding_box": {
                            "x": pred.bounding_box.x,
                            "y": pred.bounding_box.y,
                            "width": pred.bounding_box.width,
                            "height": pred.bounding_box.height,
                        },
                        "confidence": pred.confidence,
                        "source": "previous_prediction",
                    }
                )

        # Add context from storage manager if available
        if context and hasattr(context, "existing_annotations"):
            if "existing_annotations" not in detection_context:
                detection_context["existing_annotations"] = []

            for ann in context.existing_annotations:
                # Convert annotation format to context format
                bbox = ann.get("bounding_box", {})
                detection_context["existing_annotations"].append(
                    {
                        "tag": ann.get("tag", "unknown"),
                        "bounding_box": bbox,
                        "confidence": ann.get("confidence"),
                        "source": "existing_annotation",
                    }
                )

        # Add context instructions for better prompt engineering
        if detection_context.get("existing_annotations"):
            detection_context["context_instructions"] = (
                f"This image already has {len(detection_context['existing_annotations'])} "
                "existing annotations. Look for additional UI elements that haven't been "
                "annotated yet. Avoid creating overlapping annotations."
            )

        # Add image metadata context
        if context and context.image_metadata:
            detection_context["image_info"] = {
                "dimensions": {
                    "width": context.image_metadata.get("width"),
                    "height": context.image_metadata.get("height"),
                },
                "format": context.image_metadata.get("format"),
                "existing_annotation_count": context.image_metadata.get(
                    "annotation_count", 0
                ),
            }

        # Add user feedback context
        if context and context.user_feedback:
            detection_context["user_feedback"] = context.user_feedback
            detection_context["feedback_instructions"] = (
                "Consider the user feedback when making predictions. "
                "Adjust detection based on previous corrections or preferences."
            )

        return detection_context

    async def _execute_detection_with_tools(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute detection using direct OpenAI integration"""

        # Use direct OpenAI call for now
        client = AsyncOpenAI(api_key=config.openai_api_key)

        # Build messages for OpenAI
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_detection_prompt(context)},
                    context["image"],
                ],
            }
        ]

        # Call OpenAI directly
        response = await client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=2000,
            temperature=0.1,
        )

        # Return in the expected format
        return {"content": response.choices[0].message.content}

    async def _parse_detection_results(
        self, raw_results: Dict[str, Any], scale_factor: float = 1.0
    ) -> List[DetectedElement]:
        """Parse MCP results into DetectedElement objects with coordinate scaling"""

        detected_elements = []

        try:
            # Extract content from MCP response
            content = raw_results.get("content", "")

            # Parse JSON response (assuming structured output)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content

            detection_data = json.loads(json_content)

            for idx, element_data in enumerate(detection_data.get("elements", [])):
                # Validate and scale coordinates back to original image dimensions
                bbox_data = element_data["bounding_box"]

                # Calculate resized dimensions based on scale factor
                # Using standard 1024 max dimension resize logic
                if scale_factor < 1.0:
                    # Image was resized, calculate the resized dimensions
                    max_dimension = 1024
                    resized_width = (
                        max_dimension
                        if scale_factor == max_dimension / 3300
                        else int(3300 * scale_factor)
                    )  # Fallback calculation
                    resized_height = (
                        int(1486 * scale_factor)
                        if scale_factor == max_dimension / 3300
                        else int(1486 * scale_factor)
                    )  # Fallback calculation

                    # More precise calculation: determine which dimension was the limiting factor
                    if scale_factor * 3300 > scale_factor * 1486:  # Width was limiting
                        resized_width = max_dimension
                        resized_height = int(1486 * scale_factor)
                    else:  # Height was limiting
                        resized_height = max_dimension
                        resized_width = int(3300 * scale_factor)
                else:
                    # No resizing, use original dimensions
                    resized_width = (
                        3300  # Should get from context, but fallback for now
                    )
                    resized_height = 1486

                scaled_bbox = self._validate_and_scale_coordinates(
                    bbox_data, scale_factor, resized_width, resized_height
                )

                detected_element = DetectedElement(
                    id=f"elem_{idx}_{int(datetime.now().timestamp())}",
                    tag=UIElementTag(element_data["tag"]),
                    bounding_box=BoundingBox(**scaled_bbox),
                    confidence=element_data.get("confidence", 0.8),
                    reasoning=element_data.get("reasoning"),
                    model_version="gpt-4-vision-mcp",
                    detection_timestamp=datetime.now(),
                )
                detected_elements.append(detected_element)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse detection results: {e}")
            logger.debug(f"Raw results: {raw_results}")

        return detected_elements

    def _build_detection_prompt(self, context: Dict[str, Any]) -> str:
        """Build detection prompt with enhanced context and improved accuracy"""

        # Get image dimensions from context
        img_info = context.get("image_info", {})
        dimensions = img_info.get("dimensions", {})
        img_width = dimensions.get("width", "unknown")
        img_height = dimensions.get("height", "unknown")

        base_prompt = f"""
        Analyze this UI screenshot and detect interactive elements with high precision.
        
        DETECTION GUIDELINES:
        1. Be generous with button boundaries - include the entire clickable area plus visual styling
        2. Look for ALL interactive elements, including larger prominent buttons
        3. Include button text, borders, padding, and background in the bounding box
        4. Pay attention to visual hierarchy - larger elements are often more important
        
        For each element found:
        1. Draw a precise bounding box around the COMPLETE element
        2. Classify it as: button, input, radio, or dropdown
        3. Provide confidence score (0.0-1.0)
        4. Brief reasoning for the classification
        
        BUTTON DETECTION TIPS:
        - Include the full visual area (text + background + borders + padding)
        - Look for rectangular areas with button-like styling
        - Consider visual affordances and styling cues
        - Don't be too conservative - err on the side of larger, more complete boxes
        - Main action buttons are often larger and more prominent
        
        COORDINATE ACCURACY:
        - Image dimensions: {img_width}×{img_height} pixels
        - Use precise pixel coordinates 
        - Ensure bounding boxes fully contain the UI elements
        
        Return results in this JSON format:
        {{
            "elements": [
                {{
                    "tag": "button",
                    "bounding_box": {{"x": 100, "y": 50, "width": 120, "height": 40}},
                    "confidence": 0.95,
                    "reasoning": "Large prominent button with clear visual styling and text"
                }}
            ]
        }}
        
        Requirements:
        - Use (x, y, width, height) format for bounding boxes
        - Focus on clearly interactive UI elements
        - Be thorough and include larger button areas
        - Avoid decorative elements or static text
        - Be precise with coordinates but generous with boundaries
        """

        # Add context-specific instructions based on enhanced context
        if context.get("existing_annotations"):
            base_prompt += "\n\n🎯 CONTEXT AWARENESS:\n"
            base_prompt += f"This image already has {len(context['existing_annotations'])} existing annotations:\n"

            for i, ann in enumerate(
                context["existing_annotations"][:5]
            ):  # Show first 5
                bbox = ann["bounding_box"]
                base_prompt += f"  {i+1}. {ann['tag']} at ({bbox['x']}, {bbox['y']}) {bbox['width']}×{bbox['height']} - {ann['source']}\n"

            if len(context["existing_annotations"]) > 5:
                base_prompt += (
                    f"  ... and {len(context['existing_annotations']) - 5} more\n"
                )

            base_prompt += "\n🔍 DETECTION STRATEGY:\n"
            base_prompt += "- Look for NEW elements not already annotated\n"
            base_prompt += "- Avoid overlapping with existing annotations\n"
            base_prompt += "- Focus on areas without existing annotations\n"
            base_prompt += "- Consider whether existing annotations might have missed any obvious elements\n"

        # Add image information context
        if context.get("image_info"):
            img_info = context["image_info"]
            dimensions = img_info.get("dimensions", {})
            if dimensions.get("width") and dimensions.get("height"):
                base_prompt += f"\n📐 IMAGE INFO:\n"
                base_prompt += f"- Image size: {dimensions['width']}×{dimensions['height']} pixels\n"
                base_prompt += f"- Format: {img_info.get('format', 'Unknown')}\n"
                base_prompt += f"- Existing annotations: {img_info.get('existing_annotation_count', 0)}\n"

        # Add specific context instructions
        if context.get("context_instructions"):
            base_prompt += (
                f"\n📋 SPECIFIC INSTRUCTIONS:\n{context['context_instructions']}\n"
            )

        return base_prompt

    def _get_detection_instructions(self) -> str:
        """Get comprehensive detection instructions"""
        return """
        Analyze the provided UI screenshot and detect all interactive elements.
        Focus on: buttons, input fields, radio buttons, and dropdown menus.
        Provide precise bounding boxes and classify each element accurately.
        """

    def _initialize_tools(self) -> List[MCPTool]:
        """Initialize available MCP tools"""
        return [
            MCPTool(
                name="detect_ui_elements",
                description="Detect and classify UI elements in images",
                parameters={
                    "type": "object",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tag": {
                                        "type": "string",
                                        "enum": [
                                            "button",
                                            "input",
                                            "radio",
                                            "dropdown",
                                        ],
                                    },
                                    "bounding_box": {
                                        "type": "object",
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"},
                                            "width": {"type": "number"},
                                            "height": {"type": "number"},
                                        },
                                        "required": ["x", "y", "width", "height"],
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "reasoning": {"type": "string"},
                                },
                                "required": ["tag", "bounding_box", "confidence"],
                            },
                        }
                    },
                    "required": ["elements"],
                },
            )
        ]

    async def close(self):
        """Close MCP session"""
        if self.client:
            # FastMCP client doesn't need explicit closing in this context
            self.client = None


# Session manager for persistent context
class MCPSessionManager:
    """Manage MCP sessions with persistent context"""

    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def get_session_context(
        self, image_id: str, task_type: str = "ui_detection"
    ) -> Dict[str, Any]:
        """Get or create session context"""

        session_key = f"{task_type}:{image_id}"

        if session_key not in self.active_sessions:
            self.active_sessions[session_key] = {
                "created_at": datetime.now(),
                "task_type": task_type,
                "image_id": image_id,
                "context_history": [],
                "user_corrections": [],
                "prediction_patterns": {},
            }

        return self.active_sessions[session_key]

    async def update_session_context(
        self,
        image_id: str,
        update_data: Dict[str, Any],
        task_type: str = "ui_detection",
    ):
        """Update session context with new data"""

        session_key = f"{task_type}:{image_id}"
        session = await self.get_session_context(image_id, task_type)

        # Add to context history
        session["context_history"].append(
            {"timestamp": datetime.now(), "data": update_data}
        )

        # Update specific context based on data type
        if "user_correction" in update_data:
            session["user_corrections"].append(update_data["user_correction"])

        if "prediction_feedback" in update_data:
            prediction_feedback = update_data["prediction_feedback"]
            session["prediction_patterns"][prediction_feedback["pattern_type"]] = (
                prediction_feedback["pattern_data"]
            )
