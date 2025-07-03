"""
YOLO Object Detection API Route

This module provides Flask routes for YOLO-based object detection.
It handles image uploads, processes them through a YOLO model, and returns structured JSON results.
"""

import os
import io
import json
import base64
from PIL import Image
import torch
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import tempfile

# Import our YOLO to JSON converter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from yolo_to_json_converter import YOLOToJSONConverter

yolo_bp = Blueprint('yolo', __name__)

# COCO class names (commonly used with YOLO)
COCO_CLASSES = [
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

# Global variables for model and converter
model = None
converter = None

def load_model():
    """Load the YOLO model. This is called once when the Flask app starts."""
    global model, converter
    try:
        # Load YOLOv5 model from torch hub (pre-trained on COCO)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()  # Set to evaluation mode
        
        # Initialize the JSON converter
        converter = YOLOToJSONConverter(COCO_CLASSES)
        
        print("YOLO model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        return False

def process_image_with_yolo(image_path, image_name):
    """
    Process an image with YOLO and return structured JSON results.
    
    Args:
        image_path: Path to the image file
        image_name: Name of the image file
        
    Returns:
        Dictionary containing detection results in JSON format
    """
    global model, converter
    
    if model is None or converter is None:
        raise Exception("YOLO model not loaded")
    
    try:
        # Load and process the image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Run YOLO inference
        results = model(image_path)
        
        # Extract detection data
        detections = []
        if len(results.pandas().xyxy[0]) > 0:
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                detection = {
                    "class_name": row['name'],
                    "class_id": int(row['class']),
                    "confidence": round(float(row['confidence']), 3),
                    "box_2d": {
                        "x_min": int(row['xmin']),
                        "y_min": int(row['ymin']),
                        "x_max": int(row['xmax']),
                        "y_max": int(row['ymax'])
                    },
                    "normalized_box": {
                        "x_center": round((row['xmin'] + row['xmax']) / (2 * img_width), 4),
                        "y_center": round((row['ymin'] + row['ymax']) / (2 * img_height), 4),
                        "width": round((row['xmax'] - row['xmin']) / img_width, 4),
                        "height": round((row['ymax'] - row['ymin']) / img_height, 4)
                    }
                }
                detections.append(detection)
        
        # Create structured JSON output
        result = {
            "image_name": image_name,
            "image_dimensions": {
                "width": img_width,
                "height": img_height
            },
            "detections": detections,
            "detection_count": len(detections)
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@yolo_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint to verify the API is running."""
    global model
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "YOLO Object Detection API is running"
    })

@yolo_bp.route('/analyze', methods=['POST'])
@cross_origin()
def analyze_image():
    """
    Analyze an uploaded image for object detection.
    
    Accepts:
    - File upload via 'image' field
    - Base64 encoded image via JSON body with 'image_data' field
    
    Returns:
    - JSON response with detection results
    """
    try:
        image_file = None
        image_name = "uploaded_image.jpg"
        
        # Check if image is provided as file upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({"error": "No image file selected"}), 400
            image_name = secure_filename(image_file.filename)
        
        # Check if image is provided as base64 in JSON
        elif request.is_json:
            data = request.get_json()
            if 'image_data' in data:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(data['image_data'])
                    image_file = io.BytesIO(image_data)
                    image_name = data.get('image_name', 'base64_image.jpg')
                except Exception as e:
                    return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
            else:
                return jsonify({"error": "No 'image_data' field found in JSON"}), 400
        
        else:
            return jsonify({"error": "No image provided. Use file upload or base64 JSON."}), 400
        
        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            if hasattr(image_file, 'save'):
                image_file.save(temp_file.name)
            else:
                temp_file.write(image_file.read())
            temp_file_path = temp_file.name
        
        try:
            # Process the image with YOLO
            result = process_image_with_yolo(temp_file_path, image_name)
            return jsonify(result)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@yolo_bp.route('/classes', methods=['GET'])
@cross_origin()
def get_classes():
    """Get the list of classes that the YOLO model can detect."""
    return jsonify({
        "classes": COCO_CLASSES,
        "total_classes": len(COCO_CLASSES)
    })

# Initialize the model when the blueprint is imported
load_model()

