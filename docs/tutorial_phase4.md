
# Phase 4: Building a Flask API for Real-time Analysis

This phase demonstrates how to create a Flask web API that integrates YOLOv5 for real-time object detection. The API provides RESTful endpoints for image analysis and returns structured JSON results.

## 1. Designing the Flask API

Our Flask API is designed with the following key features:

*   **RESTful Endpoints:** Clean API design with proper HTTP methods and status codes
*   **Multiple Input Formats:** Support for file uploads and base64 encoded images
*   **CORS Support:** Enable cross-origin requests for web applications
*   **Performance Optimization:** Load model once at startup for faster inference
*   **Error Handling:** Robust error management with informative messages
*   **Health Checks:** Monitoring endpoints for system status

### API Endpoints

1.  **Health Check:** `GET /api/yolo/health`
    *   Returns the API status and model loading state
    *   Useful for monitoring and debugging

2.  **Analyze Image:** `POST /api/yolo/analyze`
    *   Accepts image uploads or base64 encoded images
    *   Returns structured JSON with detection results

3.  **Get Classes:** `GET /api/yolo/classes`
    *   Returns the list of classes the YOLO model can detect
    *   Useful for understanding model capabilities

## 2. Implementing the Flask Application

### Project Structure

```
yolo_detection_api/
├── venv/                    # Virtual environment
├── src/
│   ├── models/             # Database models
│   ├── routes/
│   │   ├── user.py         # User routes (from template)
│   │   └── yolo_detection.py  # YOLO detection routes
│   ├── static/
│   │   └── index.html      # Frontend interface
│   └── main.py             # Main Flask application
├── yolo_to_json_converter.py  # Our JSON converter utility
└── requirements.txt        # Python dependencies
```

### Key Implementation Details

#### 1. YOLO Detection Route (`src/routes/yolo_detection.py`)

```python
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
    # ... (full list of 80 COCO classes)
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
```

#### 2. Main Flask Application (`src/main.py`)

```python
import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.yolo_detection import yolo_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app)

app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(yolo_bp, url_prefix='/api/yolo')

# Database configuration (optional for this project)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
```

#### 3. Frontend Interface (`src/static/index.html`)

We've created a simple HTML interface that allows users to:
*   Upload images for analysis
*   View detection results in a structured format
*   Check API health status
*   View available object classes

The frontend uses JavaScript to interact with the API endpoints and displays results in a user-friendly format.

## 3. Local Testing of the Flask API

### Installation and Setup

1.  **Create the Flask application:**
    ```bash
    manus-create-flask-app yolo_detection_api
    cd yolo_detection_api
    ```

2.  **Install dependencies:**
    ```bash
    source venv/bin/activate
    pip install torch torchvision ultralytics flask-cors pillow seaborn
    pip freeze > requirements.txt
    ```

3.  **Copy the YOLO to JSON converter:**
    ```bash
    cp /path/to/yolo_to_json_converter.py ./
    ```

4.  **Create the YOLO detection route and update main.py** (as shown above)

5.  **Create the frontend interface** (as shown above)

### Running the Application

```bash
source venv/bin/activate
python src/main.py
```

The application will start on `http://localhost:5001` (or `http://0.0.0.0:5001`).

### Testing the API

You can test the API in several ways:

1.  **Using the Web Interface:**
    *   Open your browser to `http://localhost:5001`
    *   Upload an image and click "Analyze Image"
    *   View the structured JSON results

2.  **Using curl commands:**
    ```bash
    # Health check
    curl http://localhost:5001/api/yolo/health
    
    # Get classes
    curl http://localhost:5001/api/yolo/classes
    
    # Analyze image (file upload)
    curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5001/api/yolo/analyze
    ```

3.  **Using Python requests:**
    ```python
    import requests
    
    # Health check
    response = requests.get('http://localhost:5001/api/yolo/health')
    print(response.json())
    
    # Analyze image
    with open('path/to/your/image.jpg', 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:5001/api/yolo/analyze', files=files)
        print(response.json())
    ```

### Example API Response

When you upload an image with detectable objects, you'll receive a response like this:

```json
{
  "image_name": "example.jpg",
  "image_dimensions": {
    "width": 640,
    "height": 480
  },
  "detections": [
    {
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.95,
      "box_2d": {
        "x_min": 100,
        "y_min": 50,
        "x_max": 200,
        "y_max": 300
      },
      "normalized_box": {
        "x_center": 0.234,
        "y_center": 0.365,
        "width": 0.156,
        "height": 0.521
      }
    }
  ],
  "detection_count": 1
}
```

This structured JSON output makes it easy to integrate the YOLO results into other applications, databases, or visualization tools.

## Key Features Implemented

1.  **Model Loading:** The YOLO model is loaded once at startup to improve performance
2.  **Multiple Input Formats:** Support for both file uploads and base64 encoded images
3.  **Error Handling:** Comprehensive error handling with informative messages
4.  **CORS Support:** Enables cross-origin requests for web applications
5.  **Structured Output:** Uses our `yolo_to_json_converter.py` for consistent JSON formatting
6.  **Health Monitoring:** Health check endpoint for system monitoring
7.  **Clean Architecture:** Separation of concerns with blueprints and modular design

This Flask API serves as a solid foundation for real-time object detection applications and can be easily extended or integrated into larger systems.

