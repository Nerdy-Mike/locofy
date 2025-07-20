import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from PIL import Image

from models.annotation_models import Annotation, ImageMetadata, LLMPrediction


class FileStorageManager:
    """Manages file storage for images, annotations, and metadata"""

    def __init__(self, base_data_dir: str = "data"):
        self.base_dir = Path(base_data_dir)
        self.images_dir = self.base_dir / "images"
        self.annotations_dir = self.base_dir / "annotations"
        self.metadata_dir = self.base_dir / "metadata"
        self.predictions_dir = self.base_dir / "predictions"

        # Create directories if they don't exist
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for file storage"""
        for directory in [
            self.images_dir,
            self.annotations_dir,
            self.metadata_dir,
            self.predictions_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_image(self, image_file, filename: str) -> ImageMetadata:
        """Save uploaded image and return metadata"""
        # Generate unique image ID
        image_id = str(uuid4())

        # Get file extension
        file_ext = Path(filename).suffix.lower()
        if not file_ext:
            file_ext = ".jpg"  # default extension

        # Create storage filename
        storage_filename = f"{image_id}{file_ext}"
        storage_path = self.images_dir / storage_filename

        # Save the image file
        if hasattr(image_file, "save"):
            # Streamlit UploadedFile object
            with open(storage_path, "wb") as f:
                f.write(image_file.getbuffer())
        else:
            # File-like object
            shutil.copyfileobj(image_file, open(storage_path, "wb"))

        # Get image information using PIL
        with Image.open(storage_path) as img:
            width, height = img.size
            format_name = img.format

        # Create metadata
        metadata = ImageMetadata(
            id=image_id,
            filename=filename,
            file_path=str(storage_path),
            file_size=storage_path.stat().st_size,
            width=width,
            height=height,
            format=format_name or "UNKNOWN",
        )

        # Save metadata
        self.save_image_metadata(metadata)

        return metadata

    def save_image_metadata(self, metadata: ImageMetadata):
        """Save image metadata to JSON file"""
        metadata_path = self.metadata_dir / f"{metadata.id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.dict(), f, indent=2)

    def get_image_metadata(self, image_id: str) -> Optional[ImageMetadata]:
        """Load image metadata from JSON file"""
        metadata_path = self.metadata_dir / f"{image_id}.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return ImageMetadata(**data)

    def get_image_path(self, image_id: str) -> Optional[Path]:
        """Get the file path for an image"""
        metadata = self.get_image_metadata(image_id)
        if metadata:
            return Path(metadata.file_path)
        return None

    def list_images(self) -> List[ImageMetadata]:
        """List all available images"""
        images = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                images.append(ImageMetadata(**data))
            except Exception as e:
                print(f"Error loading metadata {metadata_file}: {e}")

        return sorted(images, key=lambda x: x.upload_time, reverse=True)

    def save_annotations(self, image_id: str, annotations: List[Annotation]):
        """Save annotations for an image"""
        annotations_path = self.annotations_dir / f"{image_id}.json"
        annotations_data = [ann.dict() for ann in annotations]

        with open(annotations_path, "w") as f:
            json.dump(annotations_data, f, indent=2)

        # Update annotation count in metadata
        metadata = self.get_image_metadata(image_id)
        if metadata:
            metadata.annotation_count = len(annotations)
            self.save_image_metadata(metadata)

    def get_annotations(self, image_id: str) -> List[Annotation]:
        """Load annotations for an image"""
        annotations_path = self.annotations_dir / f"{image_id}.json"
        if not annotations_path.exists():
            return []

        with open(annotations_path, "r") as f:
            data = json.load(f)

        return [Annotation(**ann_data) for ann_data in data]

    def save_llm_predictions(self, prediction: LLMPrediction):
        """Save LLM predictions for an image"""
        predictions_path = self.predictions_dir / f"{prediction.image_id}.json"

        with open(predictions_path, "w") as f:
            json.dump(prediction.dict(), f, indent=2)

        # Update metadata to indicate AI predictions exist
        metadata = self.get_image_metadata(prediction.image_id)
        if metadata:
            metadata.has_ai_predictions = True
            self.save_image_metadata(metadata)

    def get_llm_predictions(self, image_id: str) -> Optional[LLMPrediction]:
        """Load LLM predictions for an image"""
        predictions_path = self.predictions_dir / f"{image_id}.json"
        if not predictions_path.exists():
            return None

        with open(predictions_path, "r") as f:
            data = json.load(f)

        return LLMPrediction(**data)

    def delete_image(self, image_id: str) -> bool:
        """Delete an image and all associated data"""
        try:
            # Delete image file
            image_path = self.get_image_path(image_id)
            if image_path and image_path.exists():
                image_path.unlink()

            # Delete metadata
            metadata_path = self.metadata_dir / f"{image_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()

            # Delete annotations
            annotations_path = self.annotations_dir / f"{image_id}.json"
            if annotations_path.exists():
                annotations_path.unlink()

            # Delete predictions
            predictions_path = self.predictions_dir / f"{image_id}.json"
            if predictions_path.exists():
                predictions_path.unlink()

            return True
        except Exception as e:
            print(f"Error deleting image {image_id}: {e}")
            return False

    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        total_images = len(list(self.metadata_dir.glob("*.json")))
        total_annotations = len(list(self.annotations_dir.glob("*.json")))
        total_predictions = len(list(self.predictions_dir.glob("*.json")))

        # Calculate total storage size
        total_size = 0
        for directory in [
            self.images_dir,
            self.annotations_dir,
            self.metadata_dir,
            self.predictions_dir,
        ]:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "total_predictions": total_predictions,
            "total_storage_bytes": total_size,
            "total_storage_mb": round(total_size / (1024 * 1024), 2),
        }
