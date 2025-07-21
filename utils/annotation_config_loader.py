"""
Annotation Configuration Loader

Loads and manages UI element types and annotation configuration from JSON files.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


class AnnotationConfigLoader:
    """Loads and provides access to annotation configuration"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to the config file in utils directory
            config_path = Path(__file__).parent / "annotation_config.json"

        self.config_path = Path(config_path)
        self._config = None

    @property
    def config(self) -> Dict:
        """Lazy load configuration"""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, "r") as f:
                content = f.read()
                # Since the file contains JSON content, parse it
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Fallback to basic configuration
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict:
        """Fallback configuration if JSON file is not available"""
        return {
            "ui_element_types": {
                "button": {
                    "display_name": "Button",
                    "description": "Clickable button elements",
                    "color": "#007bff",
                    "manual_suggestions": [
                        "Login Button",
                        "Submit Button",
                        "Cancel Button",
                    ],
                },
                "input": {
                    "display_name": "Input Field",
                    "description": "Text input and form fields",
                    "color": "#28a745",
                    "manual_suggestions": [
                        "Email Input",
                        "Password Input",
                        "Username Input",
                    ],
                },
                "radio": {
                    "display_name": "Radio Button",
                    "description": "Single-select radio button groups",
                    "color": "#ffc107",
                    "manual_suggestions": [
                        "Gender Selection",
                        "Payment Method",
                        "Size Option",
                    ],
                },
                "dropdown": {
                    "display_name": "Dropdown Menu",
                    "description": "Dropdown select menus",
                    "color": "#6f42c1",
                    "manual_suggestions": [
                        "Country Dropdown",
                        "State Dropdown",
                        "Category Dropdown",
                    ],
                },
            },
            "validation_rules": {
                "min_box_width": 10,
                "min_box_height": 10,
                "overlap_threshold": 0.3,
                "max_annotations_per_batch": 50,
            },
            "display_settings": {
                "canvas_colors": {
                    "draft": "#007bff",
                    "tagged": "#28a745",
                    "existing": "#6c757d",
                    "temporary": "#ffc107",
                }
            },
        }

    def get_ui_element_types(self) -> Dict[str, Dict]:
        """Get all UI element type configurations"""
        return self.config.get("ui_element_types", {})

    def get_ui_element_list(self) -> List[str]:
        """Get list of UI element type keys"""
        return list(self.get_ui_element_types().keys())

    def get_ui_element_display_names(self) -> Dict[str, str]:
        """Get mapping of keys to display names"""
        types = self.get_ui_element_types()
        return {
            key: config.get("display_name", key.title())
            for key, config in types.items()
        }

    def get_manual_suggestions(self, ui_element_type: str) -> List[str]:
        """Get manual name suggestions for a specific UI element type"""
        types = self.get_ui_element_types()
        if ui_element_type in types:
            return types[ui_element_type].get("manual_suggestions", [])
        return []

    def get_element_color(self, ui_element_type: str) -> str:
        """Get color for a specific UI element type"""
        types = self.get_ui_element_types()
        if ui_element_type in types:
            return types[ui_element_type].get("color", "#6c757d")
        return "#6c757d"

    def get_validation_rules(self) -> Dict:
        """Get validation rules configuration"""
        return self.config.get("validation_rules", {})

    def get_display_settings(self) -> Dict:
        """Get display settings configuration"""
        return self.config.get("display_settings", {})

    def get_canvas_colors(self) -> Dict[str, str]:
        """Get canvas color configuration"""
        display_settings = self.get_display_settings()
        return display_settings.get(
            "canvas_colors",
            {
                "draft": "#007bff",
                "tagged": "#28a745",
                "existing": "#6c757d",
                "temporary": "#ffc107",
            },
        )


# Global instance for easy access
@lru_cache(maxsize=1)
def get_annotation_config() -> AnnotationConfigLoader:
    """Get cached annotation configuration loader"""
    return AnnotationConfigLoader()


# Convenience functions
def get_ui_element_types() -> Dict[str, Dict]:
    """Get all UI element type configurations"""
    return get_annotation_config().get_ui_element_types()


def get_ui_element_list() -> List[str]:
    """Get list of UI element type keys"""
    return get_annotation_config().get_ui_element_list()


def get_ui_element_display_names() -> Dict[str, str]:
    """Get mapping of keys to display names"""
    return get_annotation_config().get_ui_element_display_names()


def get_manual_suggestions(ui_element_type: str) -> List[str]:
    """Get manual name suggestions for a specific UI element type"""
    return get_annotation_config().get_manual_suggestions(ui_element_type)


def get_element_color(ui_element_type: str) -> str:
    """Get color for a specific UI element type"""
    return get_annotation_config().get_element_color(ui_element_type)
