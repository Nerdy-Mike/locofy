import asyncio
import base64
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from models.validation_models import (
    UIValidationRequest,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from services.llm_service import LLMUIDetectionService

# Configure logging
logger = logging.getLogger(__name__)


class UIImageValidationService:
    """Service for validating UI images using LLM before storage"""

    def __init__(
        self,
        llm_service: LLMUIDetectionService,
        config: Optional[ValidationConfig] = None,
    ):
        """
        Initialize the UI validation service

        Args:
            llm_service: Existing LLM service instance
            config: Validation configuration (uses defaults if None)
        """
        self.llm_service = llm_service
        self.config = config or ValidationConfig()

        # Cache for validation results (by image hash)
        self._validation_cache: Dict[str, ValidationResult] = {}

        logger.info(f"UIImageValidationService initialized with config: {self.config}")

    def _resize_image_for_llm(self, image_path: str) -> str:
        """
        Resize image if it's too large for efficient LLM processing

        Args:
            image_path: Path to the original image

        Returns:
            str: Path to resized image (may be same as input if no resize needed)
        """
        try:
            # Check file size first
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

            if file_size_mb <= self.config.max_image_size_mb:
                return image_path

            # Open and resize image
            with Image.open(image_path) as img:
                # Calculate new dimensions (max 1024px on longest side)
                max_dimension = 1024
                width, height = img.size

                if max(width, height) <= max_dimension:
                    return image_path

                # Calculate resize ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))

                # Resize image
                resized_img = img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

                # Save to temporary file
                resized_path = f"{image_path}_resized.jpg"
                resized_img.convert("RGB").save(resized_path, "JPEG", quality=85)

                logger.info(
                    f"Resized image from {width}x{height} to {new_width}x{new_height} "
                    f"for LLM processing: {resized_path}"
                )

                return resized_path

        except Exception as e:
            logger.warning(f"Failed to resize image {image_path}: {e}")
            return image_path  # Return original on error

    def _cleanup_temp_resized_image(self, resized_path: str, original_path: str):
        """Clean up temporary resized image if it was created"""
        if resized_path != original_path and os.path.exists(resized_path):
            try:
                os.unlink(resized_path)
                logger.debug(f"Cleaned up temporary resized image: {resized_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup resized image {resized_path}: {e}")

    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash of image for caching purposes"""
        import hashlib

        try:
            with open(image_path, "rb") as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate image hash: {e}")
            return ""

    async def validate_web_ui_image(
        self, image_path: str, request: Optional[UIValidationRequest] = None
    ) -> ValidationResult:
        """
        Validate if image contains web UI components using LLM

        Args:
            image_path: Path to the image file to validate
            request: Optional validation request with custom parameters

        Returns:
            ValidationResult: Complete validation result
        """
        start_time = time.time()

        # Use provided request or create default
        if request is None:
            request = UIValidationRequest(image_path=image_path)

        # Check if validation is enabled
        if not self.config.enabled:
            logger.info("LLM validation is disabled, allowing upload")
            return ValidationResult(
                valid=True,
                confidence=1.0,
                reason="LLM validation disabled in configuration",
                processing_time=time.time() - start_time,
                status=ValidationStatus.VALID,
                llm_model="disabled",
            )

        # Check cache if enabled
        image_hash = ""
        if self.config.cache_validation_results:
            image_hash = self._calculate_image_hash(image_path)
            if image_hash and image_hash in self._validation_cache:
                cached_result = self._validation_cache[image_hash]
                logger.info(
                    f"Using cached validation result for image hash: {image_hash}"
                )
                return cached_result

        resized_path = None

        try:
            # Validate image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Resize image if needed for LLM processing
            resized_path = self._resize_image_for_llm(image_path)

            # Encode image for LLM
            base64_image = self.llm_service.encode_image(resized_path)

            # Create validation prompt
            prompt = self._create_validation_prompt(request.include_element_detection)

            # Prepare LLM request with timeout
            llm_task = self._call_llm_for_validation(base64_image, prompt)

            try:
                # Execute with timeout
                llm_response = await asyncio.wait_for(
                    llm_task, timeout=request.timeout_seconds
                )
            except asyncio.TimeoutError:
                raise Exception(
                    f"LLM validation timed out after {request.timeout_seconds} seconds"
                )

            # Parse LLM response
            validation_result = self._parse_llm_response(
                llm_response, request.confidence_threshold, start_time
            )

            # Cache result if enabled and hash available
            if self.config.cache_validation_results and image_hash:
                self._validation_cache[image_hash] = validation_result

            logger.info(
                f"LLM validation completed: valid={validation_result.valid}, "
                f"confidence={validation_result.confidence:.3f}, "
                f"time={validation_result.processing_time:.2f}s"
            )

            return validation_result

        except Exception as e:
            error_message = str(e)
            logger.error(f"LLM validation failed: {error_message}")

            # Handle error based on fallback configuration and error type
            # Only allow fallback for network/timeout errors, not for model/API errors
            is_temporary_error = any(
                term in error_message.lower()
                for term in ["timeout", "connection", "network", "rate limit", "busy"]
            )

            if self.config.fallback_on_error and is_temporary_error:
                logger.warning(
                    "Allowing upload due to temporary validation error and fallback enabled"
                )
                return ValidationResult(
                    valid=True,
                    confidence=0.0,
                    reason=f"Temporary validation error, fallback enabled: {error_message}",
                    processing_time=time.time() - start_time,
                    status=ValidationStatus.ERROR,
                    llm_model=self.llm_service.llm_model,
                    error_message=error_message,
                )
            else:
                # For model errors, API errors, etc., reject the upload
                logger.error(
                    f"Rejecting upload due to validation error: {error_message}"
                )
                return ValidationResult(
                    valid=False,
                    confidence=0.0,
                    reason=f"Validation failed: {error_message}",
                    processing_time=time.time() - start_time,
                    status=ValidationStatus.ERROR,
                    llm_model=self.llm_service.llm_model,
                    error_message=error_message,
                )

        finally:
            # Cleanup temporary resized image
            if resized_path:
                self._cleanup_temp_resized_image(resized_path, image_path)

    def _create_validation_prompt(self, include_element_detection: bool = False) -> str:
        """Create the LLM prompt for image validation"""

        base_prompt = """
        Analyze this image and determine if it's a web interface, mobile app, or software UI screenshot that contains interactive elements.
        
        Look for:
        - Buttons, input fields, dropdowns, radio buttons, checkboxes
        - Navigation menus, toolbars, forms, modal dialogs
        - Layout consistent with web/app interfaces
        - Interactive UI components and controls
        - Software application interfaces (desktop or web)
        
        AVOID accepting:
        - Photos of people, landscapes, objects, or real-world scenes
        - Documents, PDFs, text-only images, articles, books
        - Drawings, illustrations, diagrams, charts without UI elements
        - Random screenshots without clear interactive elements
        - Code editors or terminal windows (unless they have clear UI chrome)
        
        Respond in this exact JSON format:
        {
            "result": "VALID" or "INVALID",
            "confidence": 0.0 to 1.0,
            "reason": "Brief explanation (max 50 words)"
        """

        if include_element_detection:
            base_prompt += """,
            "detected_elements": ["list of UI elements you can identify"]
        """

        base_prompt += """
        }
        
        Only return the JSON object, no additional text.
        """

        return base_prompt.strip()

    async def _call_llm_for_validation(self, base64_image: str, prompt: str) -> str:
        """Make the actual LLM API call for validation"""

        response = await asyncio.to_thread(
            self.llm_service.client.chat.completions.create,
            model=self.llm_service.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                                                    "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # Use high detail for better UI analysis
                        },
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()

    def _parse_llm_response(
        self, response_content: str, confidence_threshold: float, start_time: float
    ) -> ValidationResult:
        """Parse LLM response and create ValidationResult"""

        try:
            # Clean up response to extract JSON
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            # Parse JSON response
            response_data = json.loads(content)

            # Extract validation information
            result = response_data.get("result", "").upper()
            confidence = float(response_data.get("confidence", 0.0))
            reason = response_data.get("reason", "No reason provided")
            detected_elements = response_data.get("detected_elements", [])

            # Determine if valid based on result and confidence
            is_valid = result == "VALID" and confidence >= confidence_threshold

            # Determine status
            if result == "VALID":
                status = (
                    ValidationStatus.VALID if is_valid else ValidationStatus.INVALID
                )
            else:
                status = ValidationStatus.INVALID

            # Add confidence information to reason if below threshold
            if result == "VALID" and confidence < confidence_threshold:
                reason = f"{reason} (confidence {confidence:.2f} below threshold {confidence_threshold})"

            return ValidationResult(
                valid=is_valid,
                confidence=confidence,
                reason=reason,
                processing_time=time.time() - start_time,
                status=status,
                llm_model=self.llm_service.llm_model,
                detected_elements=detected_elements if detected_elements else None,
            )

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error(
                f"Failed to parse LLM response: {e}, content: {response_content}"
            )

            # Fallback: try to extract basic info from text
            is_valid = "VALID" in response_content.upper()
            confidence = 0.5 if is_valid else 0.0

            return ValidationResult(
                valid=is_valid,
                confidence=confidence,
                reason=f"Failed to parse LLM response: {str(e)}",
                processing_time=time.time() - start_time,
                status=ValidationStatus.ERROR,
                llm_model=self.llm_service.llm_model,
                error_message=f"Response parsing error: {str(e)}",
            )

    def clear_validation_cache(self):
        """Clear the validation result cache"""
        self._validation_cache.clear()
        logger.info("Validation cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get statistics about the validation cache"""
        return {
            "cache_size": len(self._validation_cache),
            "cache_enabled": self.config.cache_validation_results,
        }
