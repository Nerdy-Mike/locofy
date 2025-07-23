#!/usr/bin/env python3
"""
Test Data Generator for UI Component Annotation Evaluation

Generates realistic test datasets:
- 100 ground truth annotation files
- 100 corresponding LLM prediction files

Usage:
    python generate_test_data.py --output-dir ./test_datasets --count 100
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


class TestDataGenerator:
    """Generates realistic test annotation and prediction data"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.ui_tags = ["button", "input", "radio", "dropdown"]

        # Realistic UI component size ranges (width, height)
        self.component_sizes = {
            "button": [(80, 30), (200, 50), (120, 40), (150, 35)],
            "input": [(200, 30), (300, 40), (250, 35), (180, 32)],
            "radio": [(20, 20), (25, 25), (18, 18), (22, 22)],
            "dropdown": [(150, 30), (200, 35), (180, 32), (220, 38)],
        }

        # Typical screen sizes for UI design
        self.image_sizes = [
            (1920, 1080),  # Desktop HD
            (1366, 768),  # Laptop
            (375, 812),  # Mobile iPhone X
            (414, 896),  # Mobile iPhone XR
            (768, 1024),  # Tablet
        ]

    def generate_realistic_bounding_boxes(
        self, image_width: int, image_height: int, num_components: int = None
    ) -> List[Dict]:
        """Generate realistic bounding boxes that don't overlap significantly"""

        if num_components is None:
            num_components = random.randint(2, 8)  # 2-8 components per image

        annotations = []
        used_areas = []  # Track used areas to avoid excessive overlap

        for _ in range(num_components):
            tag = random.choice(self.ui_tags)
            attempts = 0
            max_attempts = 50

            while attempts < max_attempts:
                # Get realistic size for this component type
                width, height = random.choice(self.component_sizes[tag])

                # Add some variation (±20%)
                width = int(width * random.uniform(0.8, 1.2))
                height = int(height * random.uniform(0.8, 1.2))

                # Ensure component fits in image
                max_x = max(0, image_width - width)
                max_y = max(0, image_height - height)

                if max_x <= 0 or max_y <= 0:
                    attempts += 1
                    continue

                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

                # Check for excessive overlap with existing components
                new_area = (x, y, x + width, y + height)
                overlap_ok = True

                for used_x1, used_y1, used_x2, used_y2 in used_areas:
                    # Calculate overlap
                    overlap_x = max(
                        0, min(new_area[2], used_x2) - max(new_area[0], used_x1)
                    )
                    overlap_y = max(
                        0, min(new_area[3], used_y2) - max(new_area[1], used_y1)
                    )
                    overlap_area = overlap_x * overlap_y

                    new_component_area = width * height
                    overlap_ratio = overlap_area / new_component_area

                    # Allow small overlaps (< 30%) but avoid major ones
                    if overlap_ratio > 0.3:
                        overlap_ok = False
                        break

                if overlap_ok:
                    used_areas.append(new_area)

                    annotation = {
                        "id": str(uuid.uuid4()),
                        "image_id": None,  # Will be set later
                        "bounding_box": {
                            "x": float(x),
                            "y": float(y),
                            "width": float(width),
                            "height": float(height),
                        },
                        "tag": tag,
                        "confidence": None,
                        "annotator": "human_annotator",
                        "created_at": self._random_timestamp().isoformat(),
                        "updated_at": self._random_timestamp().isoformat(),
                        "status": "active",
                        "reviewed_by": None,
                        "reviewed_at": None,
                        "conflicts_with": [],
                        "reasoning": None,
                    }
                    annotations.append(annotation)
                    break

                attempts += 1

        return annotations

    def add_prediction_noise(self, ground_truth: Dict) -> Dict:
        """Add realistic noise to ground truth to simulate LLM predictions"""

        prediction = ground_truth.copy()
        prediction["id"] = str(uuid.uuid4())
        prediction["annotator"] = "gpt-4v"
        prediction["status"] = "draft"
        prediction["confidence"] = round(random.uniform(0.7, 0.98), 2)

        bbox = prediction["bounding_box"].copy()

        # Add coordinate noise (±5-15 pixels)
        noise_level = random.uniform(5, 15)
        bbox["x"] += random.uniform(-noise_level, noise_level)
        bbox["y"] += random.uniform(-noise_level, noise_level)

        # Add size noise (±10% to ±25%)
        size_noise = random.uniform(0.9, 1.15)
        bbox["width"] *= size_noise
        bbox["height"] *= size_noise

        # Ensure bounds are still valid
        bbox["x"] = max(0, bbox["x"])
        bbox["y"] = max(0, bbox["y"])

        prediction["bounding_box"] = bbox

        # Occasionally predict wrong tag (5% chance)
        if random.random() < 0.05:
            prediction["tag"] = random.choice(self.ui_tags)

        return prediction

    def generate_false_positives(
        self, image_width: int, image_height: int, count: int = None
    ) -> List[Dict]:
        """Generate false positive predictions (elements that don't exist in ground truth)"""

        if count is None:
            count = random.randint(0, 2)  # 0-2 false positives per image

        false_positives = []

        for _ in range(count):
            tag = random.choice(self.ui_tags)
            width, height = random.choice(self.component_sizes[tag])

            # Add variation
            width = int(width * random.uniform(0.8, 1.2))
            height = int(height * random.uniform(0.8, 1.2))

            max_x = max(0, image_width - width)
            max_y = max(0, image_height - height)

            if max_x > 0 and max_y > 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

                false_positive = {
                    "id": str(uuid.uuid4()),
                    "image_id": None,
                    "bounding_box": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height),
                    },
                    "tag": tag,
                    "confidence": round(random.uniform(0.6, 0.85), 2),
                    "annotator": "gpt-4v",
                    "created_at": self._random_timestamp().isoformat(),
                    "updated_at": self._random_timestamp().isoformat(),
                    "status": "draft",
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "conflicts_with": [],
                    "reasoning": None,
                }
                false_positives.append(false_positive)

        return false_positives

    def _random_timestamp(self) -> datetime:
        """Generate a random timestamp within the last 30 days"""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        return start + timedelta(
            seconds=random.randint(0, int((end - start).total_seconds()))
        )

    def generate_dataset(self, output_dir: Path, count: int = 100):
        """Generate complete test dataset"""

        ground_truth_dir = output_dir / "ground_truth"
        predictions_dir = output_dir / "predictions"

        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Generating {count} test annotation pairs...")

        stats = {
            "total_annotations": 0,
            "total_predictions": 0,
            "annotations_by_tag": {tag: 0 for tag in self.ui_tags},
            "predictions_by_tag": {tag: 0 for tag in self.ui_tags},
        }

        for i in range(count):
            image_id = str(uuid.uuid4())
            image_width, image_height = random.choice(self.image_sizes)

            # Generate ground truth annotations
            annotations = self.generate_realistic_bounding_boxes(
                image_width, image_height
            )

            # Set image_id for all annotations
            for ann in annotations:
                ann["image_id"] = image_id
                stats["annotations_by_tag"][ann["tag"]] += 1

            stats["total_annotations"] += len(annotations)

            # Generate predictions (mix of true positives and false positives)
            predictions = []

            # Convert some ground truth to predictions (true positives with noise)
            for ann in annotations:
                # 80% chance to predict each ground truth annotation
                if random.random() < 0.8:
                    pred = self.add_prediction_noise(ann)
                    predictions.append(pred)
                    stats["predictions_by_tag"][pred["tag"]] += 1

            # Add some false positives
            false_positives = self.generate_false_positives(image_width, image_height)
            for fp in false_positives:
                fp["image_id"] = image_id
                stats["predictions_by_tag"][fp["tag"]] += 1

            predictions.extend(false_positives)
            stats["total_predictions"] += len(predictions)

            # Save ground truth file
            gt_file = ground_truth_dir / f"{image_id}.json"
            with open(gt_file, "w") as f:
                json.dump(annotations, f, indent=2)

            # Save predictions file (in the wrapper format)
            pred_file = predictions_dir / f"{image_id}.json"
            predictions_data = {
                "image_id": image_id,
                "predictions": predictions,
                "llm_model": "gpt-4o",
                "processing_time": round(random.uniform(2.0, 8.0), 2),
                "created_at": self._random_timestamp().isoformat(),
            }

            with open(pred_file, "w") as f:
                json.dump(predictions_data, f, indent=2)

            if (i + 1) % 10 == 0:
                print(f"✅ Generated {i + 1}/{count} annotation pairs")

        # Save dataset statistics
        stats_file = output_dir / "dataset_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n📊 Dataset Generation Complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Total annotations: {stats['total_annotations']}")
        print(f"🤖 Total predictions: {stats['total_predictions']}")
        print(f"📈 Annotations by tag: {stats['annotations_by_tag']}")
        print(f"🎯 Predictions by tag: {stats['predictions_by_tag']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for annotation evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_datasets",
        help="Output directory for test data",
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of annotation pairs to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible results"
    )

    args = parser.parse_args()

    generator = TestDataGenerator(seed=args.seed)
    output_path = Path(args.output_dir)

    generator.generate_dataset(output_path, args.count)


if __name__ == "__main__":
    main()
