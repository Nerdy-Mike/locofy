import base64
import time
from typing import List, Optional

import openai
from openai import OpenAI

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
        self.llm_model = "gpt-4-vision-preview"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def detect_ui_components(self, image_id: str, image_path: str) -> LLMPrediction:
        """Detect UI components in an image using GPT-4V"""
        start_time = time.time()

        # Encode image
        base64_image = self.encode_image(image_path)

        # Create the prompt for UI component detection
        prompt = """
        You are an expert UI/UX analyst. Analyze this UI screenshot and identify all interactive UI components.
        
        For each UI component you find, provide:
        1. The bounding box coordinates (x, y, width, height) where (0,0) is top-left
        2. The component type: button, input, radio, or dropdown
        3. A confidence score between 0.0 and 1.0
        
        Focus on clearly visible interactive elements. Ignore decorative elements, text labels, or static content.
        
        Return your response as a JSON array with this exact format:
        [
          {
            "type": "button|input|radio|dropdown",
            "x": number,
            "y": number, 
            "width": number,
            "height": number,
            "confidence": number
          }
        ]
        
        Only return the JSON array, no additional text.
        """

        try:
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

            # Convert to Annotation objects
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

                    # Create bounding box
                    bbox = BoundingBox(
                        x=float(detection["x"]),
                        y=float(detection["y"]),
                        width=float(detection["width"]),
                        height=float(detection["height"]),
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
