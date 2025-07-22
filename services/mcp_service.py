"""
MCP (Model Context Protocol) service for enhanced LLM integration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    """Enhanced UI detection service using Model Context Protocol"""

    def __init__(self):
        self.client: Optional[Any] = None  # Placeholder for future MCP client
        self.tools_registry = self._initialize_tools()

    async def initialize(self):
        """Initialize MCP connection"""
        try:
            # For now, we'll use direct OpenAI integration instead of MCP
            # since the MCP OpenAI server may not be available
            logger.info(
                "MCP-style service initialized (using direct OpenAI integration)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            raise

    async def detect_ui_elements(
        self, image_data: bytes, image_id: str, context: Optional[MCPContext] = None
    ) -> PredictionResponse:
        """Main entry point for UI element detection using MCP"""

        if not self.client:
            await self.initialize()

        start_time = datetime.now()

        try:
            # Build detection context
            detection_context = await self._build_detection_context(
                image_data, image_id, context
            )

            # Execute UI detection with MCP tools
            raw_results = await self._execute_detection_with_tools(detection_context)

            # Parse and validate results
            detected_elements = await self._parse_detection_results(raw_results)

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
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
            "task": "ui_element_detection",
            "instructions": self._get_detection_instructions(),
            "ui_element_types": ["button", "input", "radio", "dropdown"],
        }

        # Enhanced context building with existing annotations
        if context:
            # Add existing annotations if available
            existing_annotations = []
            if context.previous_predictions:
                # Convert DetectedElements to annotation format
                for elem in context.previous_predictions:
                    annotation_data = {
                        "id": elem.id,
                        "tag": elem.tag,
                        "bounding_box": elem.bounding_box,
                        "confidence": elem.confidence,
                        "source": "previous_prediction",
                    }
                    existing_annotations.append(annotation_data)

            # If we have image metadata, we can load actual annotations from storage
            if (
                context.image_metadata
                and context.image_metadata.get("annotation_count", 0) > 0
            ):
                try:
                    # Try to load actual annotations from storage
                    from utils.file_storage import FileStorageManager

                    storage_manager = FileStorageManager()
                    actual_annotations = storage_manager.get_annotations(image_id)

                    for ann in actual_annotations:
                        annotation_data = {
                            "id": ann.id,
                            "tag": ann.tag.value,
                            "bounding_box": {
                                "x": ann.bounding_box.x,
                                "y": ann.bounding_box.y,
                                "width": ann.bounding_box.width,
                                "height": ann.bounding_box.height,
                            },
                            "confidence": ann.confidence or 1.0,
                            "source": "manual_annotation",
                            "status": ann.status.value,
                            "annotator": ann.annotator,
                        }
                        existing_annotations.append(annotation_data)

                except Exception as e:
                    logger.warning(
                        f"Could not load existing annotations for context: {e}"
                    )

            if existing_annotations:
                detection_context["existing_annotations"] = existing_annotations
                detection_context["context_instructions"] = (
                    f"Consider {len(existing_annotations)} existing annotations when detecting new elements. "
                    "Avoid overlapping with existing annotations and look for elements that might have been missed."
                )

            # Add image metadata context
            if context.image_metadata:
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
            if context.user_feedback:
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
        from openai import AsyncOpenAI

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
        self, raw_results: Dict[str, Any]
    ) -> List[DetectedElement]:
        """Parse MCP results into DetectedElement objects"""

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
                detected_element = DetectedElement(
                    id=f"elem_{idx}_{int(datetime.now().timestamp())}",
                    tag=UIElementTag(element_data["tag"]),
                    bounding_box=BoundingBox(**element_data["bounding_box"]),
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
        """Build detection prompt with enhanced context"""

        base_prompt = """
        Analyze this UI screenshot and detect interactive elements. For each element found:
        
        1. Draw a precise bounding box around it
        2. Classify it as: button, input, radio, or dropdown
        3. Provide confidence score (0.0-1.0)
        4. Brief reasoning for the classification
        
        Return results in this JSON format:
        {
            "elements": [
                {
                    "tag": "button",
                    "bounding_box": {"x": 100, "y": 50, "width": 120, "height": 40},
                    "confidence": 0.95,
                    "reasoning": "Blue rectangular element with text that looks clickable"
                }
            ]
        }
        
        Requirements:
        - Use (x, y, width, height) format for bounding boxes
        - Only detect clearly interactive UI elements
        - Avoid decorative elements or static text
        - Be precise with coordinates
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

        # Add user feedback context
        if context.get("user_feedback"):
            base_prompt += "\n💬 USER FEEDBACK:\n"
            feedback = context["user_feedback"]
            if isinstance(feedback, dict):
                base_prompt += f"Previous feedback: {feedback.get('feedback_text', 'No specific feedback')}\n"
            else:
                base_prompt += f"Previous feedback: {feedback}\n"

            if context.get("feedback_instructions"):
                base_prompt += f"{context['feedback_instructions']}\n"

        # Add quality guidance
        base_prompt += """
        
        🎯 QUALITY GUIDELINES:
        - Prioritize precision over recall (better to miss an element than false positive)
        - Use confidence scores thoughtfully (0.9+ for very obvious, 0.7+ for probable, 0.5+ for uncertain)
        - Provide clear reasoning for each detection
        - Consider the overall UI pattern and element relationships
        """

        return base_prompt

    def _get_detection_instructions(self) -> str:
        """Get detailed detection instructions"""
        return """
        Detect and classify UI elements in web interfaces, mobile apps, or software screenshots.
        
        Target Elements:
        - button: Clickable buttons, submit buttons, navigation buttons
        - input: Text input fields, search boxes, text areas
        - radio: Radio button controls (circular selection)
        - dropdown: Dropdown menus, select boxes, comboboxes
        
        Guidelines:
        - Draw tight bounding boxes around visible element boundaries
        - Consider element styling and visual cues
        - Exclude pure text or decorative elements
        - Use coordinate system with origin at top-left (0,0)
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
