import json
import os
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class UILabelingAPIClient:
    """Client for interacting with UI Component Labeling API"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.base_url = self.base_url.rstrip("/")

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_health(self) -> Dict:
        """Get API health status"""
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    # Image Management Methods

    def upload_image(self, file_content: bytes, filename: str) -> Dict:
        """Upload an image file"""
        files = {"file": (filename, file_content, "image/jpeg")}
        response = self.session.post(
            f"{self.base_url}/images/upload", files=files, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def list_images(self) -> List[Dict]:
        """Get list of all uploaded images"""
        response = self.session.get(f"{self.base_url}/images", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_image_metadata(self, image_id: str) -> Dict:
        """Get metadata for a specific image"""
        response = self.session.get(f"{self.base_url}/images/{image_id}", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_image_file(self, image_id: str) -> bytes:
        """Get the actual image file content"""
        response = self.session.get(
            f"{self.base_url}/images/{image_id}/file", timeout=30
        )
        response.raise_for_status()
        return response.content

    def delete_image(self, image_id: str) -> Dict:
        """Delete an image and all associated data"""
        response = self.session.delete(f"{self.base_url}/images/{image_id}", timeout=10)
        response.raise_for_status()
        return response.json()

    # Annotation Management Methods

    def get_annotations(self, image_id: str) -> List[Dict]:
        """Get all annotations for an image"""
        response = self.session.get(
            f"{self.base_url}/annotations/{image_id}", timeout=10
        )
        response.raise_for_status()
        return response.json()

    def save_annotation_batch(
        self, image_id: str, created_by: str, annotations: List[Dict]
    ) -> Dict:
        """Save a batch of annotations for an image"""
        batch_data = {
            "image_id": image_id,
            "created_by": created_by,
            "annotations": annotations,
        }
        response = self.session.post(
            f"{self.base_url}/annotations/batch",
            json=batch_data,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_annotation_conflicts(self, image_id: str) -> Dict:
        """Get conflicts for annotations on a specific image"""
        response = self.session.get(
            f"{self.base_url}/annotations/{image_id}/conflicts", timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_annotation_statistics(self) -> Dict:
        """Get comprehensive statistics about annotations"""
        response = self.session.get(
            f"{self.base_url}/annotations/statistics", timeout=10
        )
        response.raise_for_status()
        return response.json()

    # LLM Prediction Methods

    def generate_predictions(self, image_id: str) -> Dict:
        """Generate LLM predictions for an image using direct API"""
        response = self.session.post(
            f"{self.base_url}/images/{image_id}/predict",
            timeout=60,  # Longer timeout for LLM processing
        )
        response.raise_for_status()
        return response.json()

    def generate_mcp_predictions(self, image_id: str) -> Dict:
        """Generate LLM predictions for an image using MCP with context awareness"""
        response = self.session.post(
            f"{self.base_url}/images/{image_id}/predict-mcp",
            timeout=90,  # Longer timeout for MCP processing
        )
        response.raise_for_status()
        return response.json()

    def get_predictions(self, image_id: str) -> Optional[Dict]:
        """Get existing LLM predictions for an image"""
        response = self.session.get(
            f"{self.base_url}/images/{image_id}/predictions", timeout=10
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    # Evaluation and Statistics Methods

    def evaluate_predictions(self, image_id: str) -> Dict:
        """Evaluate LLM predictions against ground truth"""
        response = self.session.get(
            f"{self.base_url}/images/{image_id}/evaluation", timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_statistics(self) -> Dict:
        """Get system-wide statistics"""
        response = self.session.get(f"{self.base_url}/statistics", timeout=10)
        response.raise_for_status()
        return response.json()

    # Utility Methods

    def is_api_available(self) -> bool:
        """Check if API is available"""
        try:
            self.get_health()
            return True
        except Exception:
            return False
