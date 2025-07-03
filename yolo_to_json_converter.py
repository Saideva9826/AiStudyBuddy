#!/usr/bin/env python3
"""
YOLO Output to JSON Converter

This script demonstrates how to convert YOLO text output format to structured JSON.
YOLO typically outputs predictions in the format:
class_id x_center y_center width height confidence

This script converts that to a more structured JSON format.
"""

import json
import os
from typing import List, Dict, Any


class YOLOToJSONConverter:
    """Converts YOLO text output to structured JSON format."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize the converter with class names.
        
        Args:
            class_names: List of class names corresponding to class IDs
        """
        self.class_names = class_names
    
    def convert_yolo_line_to_detection(self, line: str, image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Convert a single YOLO detection line to a detection dictionary.
        
        Args:
            line: YOLO format line (class_id x_center y_center width height confidence)
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            Dictionary containing detection information
        """
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO format line: {line}")
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0
        
        # Convert normalized coordinates to pixel coordinates
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        
        # Get class name
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
        
        return {
            "class_name": class_name,
            "class_id": class_id,
            "confidence": round(confidence, 3),
            "box_2d": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            },
            "normalized_box": {
                "x_center": round(x_center, 4),
                "y_center": round(y_center, 4),
                "width": round(width, 4),
                "height": round(height, 4)
            }
        }
    
    def convert_file_to_json(self, yolo_file_path: str, image_name: str, 
                           image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Convert a YOLO text file to JSON format.
        
        Args:
            yolo_file_path: Path to the YOLO text file
            image_name: Name of the corresponding image
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            Dictionary containing all detections in JSON format
        """
        detections = []
        
        if os.path.exists(yolo_file_path):
            with open(yolo_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            detection = self.convert_yolo_line_to_detection(line, image_width, image_height)
                            detections.append(detection)
                        except ValueError as e:
                            print(f"Warning: Skipping invalid line: {e}")
        
        return {
            "image_name": image_name,
            "image_dimensions": {
                "width": image_width,
                "height": image_height
            },
            "detections": detections,
            "detection_count": len(detections)
        }


def main():
    """Example usage of the YOLO to JSON converter."""
    
    # COCO dataset class names (commonly used with YOLO)
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    # Initialize converter
    converter = YOLOToJSONConverter(coco_classes)
    
    # Create example YOLO output
    example_yolo_content = """0 0.5 0.3 0.2 0.4 0.95
2 0.7 0.6 0.15 0.25 0.88
0 0.2 0.8 0.1 0.15 0.76"""
    
    # Write example YOLO file
    with open("example_detections.txt", "w") as f:
        f.write(example_yolo_content)
    
    # Convert to JSON
    result = converter.convert_file_to_json(
        "example_detections.txt", 
        "example_image.jpg", 
        640, 480
    )
    
    # Save JSON output
    with open("example_detections.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Conversion completed!")
    print("Input YOLO format:")
    print(example_yolo_content)
    print("\nOutput JSON format:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

