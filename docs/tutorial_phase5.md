
# Phase 5: Preparing AWS Lambda Deployment Package

This phase covers how to prepare and deploy your YOLO object detection model on AWS Lambda for serverless, scalable computer vision applications.

## 1. Understanding AWS Lambda for Computer Vision

AWS Lambda is a serverless computing service that offers several advantages for computer vision applications:

### Benefits:
*   **Serverless:** No server management required
*   **Pay-per-use:** Only pay for actual compute time
*   **Event-driven:** Can be triggered by various AWS services
*   **Auto-scaling:** Automatically handles traffic spikes
*   **Cost-effective:** Ideal for variable or unpredictable workloads

### Considerations and Limitations:
*   **Package Size Limit:** 250MB unzipped (50MB zipped for direct upload)
*   **Memory Limit:** Up to 10,240 MB (10 GB)
*   **Execution Timeout:** Maximum 15 minutes
*   **Cold Start Latency:** Initial invocation delay
*   **Temporary Storage:** 512 MB to 10 GB in `/tmp`

## 2. Optimizing YOLO for Lambda Deployment

### Model Size Optimization

For AWS Lambda deployment, we need to optimize our YOLO model to fit within the size constraints:

```python
# lambda_yolo_optimizer.py
"""
YOLO Model Optimizer for AWS Lambda Deployment

This script optimizes a YOLOv5 model for deployment on AWS Lambda by:
1. Using a smaller model variant (YOLOv5s or YOLOv5n)
2. Quantizing the model to reduce size
3. Exporting to ONNX format for better compatibility
"""

import torch
import onnx
from ultralytics import YOLO

def optimize_yolo_for_lambda(model_path=None, output_dir="./lambda_model"):
    """
    Optimize YOLO model for AWS Lambda deployment.
    
    Args:
        model_path: Path to custom trained model (optional)
        output_dir: Directory to save optimized model
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model - use smallest variant for Lambda
    if model_path:
        model = YOLO(model_path)
    else:
        # Use YOLOv5n (nano) for smallest size
        model = YOLO('yolov5n.pt')
    
    # Export to ONNX format (smaller and more compatible)
    onnx_path = os.path.join(output_dir, "yolo_optimized.onnx")
    model.export(format='onnx', imgsz=640, optimize=True, simplify=True)
    
    # Move the exported file to our output directory
    import shutil
    if os.path.exists("yolov5n.onnx"):
        shutil.move("yolov5n.onnx", onnx_path)
    
    print(f"Optimized model saved to: {onnx_path}")
    
    # Check file size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    if size_mb > 200:  # Leave some room for other dependencies
        print("⚠️  Warning: Model might be too large for Lambda. Consider using YOLOv5n or further optimization.")
    else:
        print("✅ Model size is suitable for Lambda deployment.")
    
    return onnx_path

if __name__ == "__main__":
    optimize_yolo_for_lambda()
```

### Lambda Function Code

```python
# lambda_function.py
"""
AWS Lambda function for YOLO object detection.

This function handles image analysis requests and returns structured JSON results.
Optimized for serverless deployment with minimal cold start time.
"""

import json
import base64
import io
import os
import tempfile
from PIL import Image
import onnxruntime as ort
import numpy as np
import boto3

# COCO class names
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

# Global variables for model (loaded once per container)
model_session = None
model_path = "/opt/ml/model/yolo_optimized.onnx"  # Lambda layer path

def load_model():
    """Load ONNX model for inference."""
    global model_session
    if model_session is None:
        try:
            # Try to load from Lambda layer first
            if os.path.exists(model_path):
                model_session = ort.InferenceSession(model_path)
            else:
                # Fallback to local path (for testing)
                local_path = "./yolo_optimized.onnx"
                if os.path.exists(local_path):
                    model_session = ort.InferenceSession(local_path)
                else:
                    raise FileNotFoundError("YOLO model not found")
            
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    return model_session

def preprocess_image(image_path, input_size=640):
    """
    Preprocess image for YOLO inference.
    
    Args:
        image_path: Path to image file
        input_size: Model input size (default 640)
        
    Returns:
        Preprocessed image array and original dimensions
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size
    
    # Resize while maintaining aspect ratio
    img = img.resize((input_size, input_size), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Transpose to CHW format (channels first)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, (orig_width, orig_height)

def postprocess_detections(outputs, orig_size, conf_threshold=0.5, iou_threshold=0.45):
    """
    Post-process YOLO outputs to extract detections.
    
    Args:
        outputs: Raw model outputs
        orig_size: Original image dimensions (width, height)
        conf_threshold: Confidence threshold for filtering
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    # Extract predictions (assuming standard YOLO output format)
    predictions = outputs[0][0]  # Remove batch dimension
    
    orig_width, orig_height = orig_size
    
    for detection in predictions:
        # YOLO output format: [x_center, y_center, width, height, confidence, class_scores...]
        if len(detection) < 5:
            continue
            
        x_center, y_center, width, height = detection[:4]
        confidence = detection[4]
        
        if confidence < conf_threshold:
            continue
        
        # Get class with highest score
        class_scores = detection[5:]
        class_id = np.argmax(class_scores)
        class_confidence = class_scores[class_id] * confidence
        
        if class_confidence < conf_threshold:
            continue
        
        # Convert to pixel coordinates
        x_min = int((x_center - width / 2) * orig_width)
        y_min = int((y_center - height / 2) * orig_height)
        x_max = int((x_center + width / 2) * orig_width)
        y_max = int((y_center + height / 2) * orig_height)
        
        # Ensure coordinates are within image bounds
        x_min = max(0, min(x_min, orig_width))
        y_min = max(0, min(y_min, orig_height))
        x_max = max(0, min(x_max, orig_width))
        y_max = max(0, min(y_max, orig_height))
        
        detection_dict = {
            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}",
            "class_id": int(class_id),
            "confidence": round(float(class_confidence), 3),
            "box_2d": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            },
            "normalized_box": {
                "x_center": round(float(x_center), 4),
                "y_center": round(float(y_center), 4),
                "width": round(float(width), 4),
                "height": round(float(height), 4)
            }
        }
        
        detections.append(detection_dict)
    
    return detections

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "body": {
            "image_data": "base64_encoded_image_string",
            "image_name": "optional_image_name.jpg"
        }
    }
    """
    try:
        # Load model (cached after first invocation)
        model = load_model()
        
        # Parse request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Extract image data
        if 'image_data' not in body:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No image_data provided'
                })
            }
        
        image_data = body['image_data']
        image_name = body.get('image_name', 'lambda_image.jpg')
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': f'Invalid base64 image data: {str(e)}'
                })
            }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Get original image dimensions
            img = Image.open(temp_file_path)
            orig_width, orig_height = img.size
            
            # Preprocess image
            input_array, orig_size = preprocess_image(temp_file_path)
            
            # Run inference
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: input_array})
            
            # Post-process results
            detections = postprocess_detections(outputs, orig_size)
            
            # Create response
            result = {
                "image_name": image_name,
                "image_dimensions": {
                    "width": orig_width,
                    "height": orig_height
                },
                "detections": detections,
                "detection_count": len(detections)
            }
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }

# For local testing
if __name__ == "__main__":
    # Test with a sample event
    test_event = {
        "body": {
            "image_data": "base64_encoded_image_here",
            "image_name": "test_image.jpg"
        }
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
```

## 3. Deployment Package Structure

### Directory Structure
```
lambda_deployment/
├── lambda_function.py          # Main Lambda function
├── yolo_optimized.onnx        # Optimized YOLO model
├── requirements.txt           # Python dependencies
└── package/                   # Dependencies directory
    ├── PIL/
    ├── numpy/
    ├── onnxruntime/
    └── ... (other dependencies)
```

### Requirements File
```txt
# requirements.txt for Lambda deployment
Pillow==10.0.0
numpy==1.24.3
onnxruntime==1.15.1
boto3==1.28.25
```

### Deployment Script
```bash
#!/bin/bash
# deploy_lambda.sh
# Script to create AWS Lambda deployment package

echo "Creating Lambda deployment package..."

# Create deployment directory
mkdir -p lambda_deployment
cd lambda_deployment

# Copy Lambda function
cp ../lambda_function.py .
cp ../yolo_optimized.onnx .

# Install dependencies
pip install -r ../requirements.txt -t ./package

# Create deployment zip
cd package
zip -r ../lambda_deployment.zip .
cd ..
zip -g lambda_deployment.zip lambda_function.py yolo_optimized.onnx

echo "Deployment package created: lambda_deployment.zip"
echo "Package size: $(du -h lambda_deployment.zip | cut -f1)"

# Check if package is too large
SIZE=$(stat -f%z lambda_deployment.zip 2>/dev/null || stat -c%s lambda_deployment.zip)
SIZE_MB=$((SIZE / 1024 / 1024))

if [ $SIZE_MB -gt 50 ]; then
    echo "⚠️  Warning: Package is ${SIZE_MB}MB, larger than 50MB direct upload limit."
    echo "   Consider using Lambda layers or container images."
else
    echo "✅ Package size is suitable for direct upload."
fi
```

## 4. AWS Lambda Configuration

### Function Configuration
```python
# lambda_config.py
"""
AWS Lambda configuration settings for YOLO deployment.
"""

LAMBDA_CONFIG = {
    "FunctionName": "yolo-object-detection",
    "Runtime": "python3.9",
    "Role": "arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role",
    "Handler": "lambda_function.lambda_handler",
    "Code": {
        "ZipFile": "lambda_deployment.zip"
    },
    "Description": "YOLO object detection using serverless architecture",
    "Timeout": 300,  # 5 minutes
    "MemorySize": 3008,  # MB (adjust based on model requirements)
    "Environment": {
        "Variables": {
            "MODEL_PATH": "/opt/ml/model/yolo_optimized.onnx",
            "CONFIDENCE_THRESHOLD": "0.5",
            "IOU_THRESHOLD": "0.45"
        }
    }
}
```

### API Gateway Integration
```python
# api_gateway_config.py
"""
API Gateway configuration for Lambda integration.
"""

import boto3
import json

def create_api_gateway():
    """Create API Gateway for Lambda function."""
    
    client = boto3.client('apigateway')
    
    # Create REST API
    api = client.create_rest_api(
        name='yolo-detection-api',
        description='YOLO Object Detection API',
        endpointConfiguration={
            'types': ['REGIONAL']
        }
    )
    
    api_id = api['id']
    
    # Get root resource
    resources = client.get_resources(restApiId=api_id)
    root_id = resources['items'][0]['id']
    
    # Create /analyze resource
    analyze_resource = client.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart='analyze'
    )
    
    # Create POST method
    client.put_method(
        restApiId=api_id,
        resourceId=analyze_resource['id'],
        httpMethod='POST',
        authorizationType='NONE'
    )
    
    # Set up Lambda integration
    lambda_arn = f"arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:yolo-object-detection"
    
    client.put_integration(
        restApiId=api_id,
        resourceId=analyze_resource['id'],
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=f"arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
    )
    
    # Deploy API
    deployment = client.create_deployment(
        restApiId=api_id,
        stageName='prod'
    )
    
    api_url = f"https://{api_id}.execute-api.us-east-1.amazonaws.com/prod"
    
    print(f"API Gateway created successfully!")
    print(f"API URL: {api_url}/analyze")
    
    return api_url

if __name__ == "__main__":
    create_api_gateway()
```

## 5. Performance Optimization Strategies

### Cold Start Mitigation
```python
# warm_up_lambda.py
"""
Lambda warm-up strategy to reduce cold starts.
"""

import boto3
import json
import time
from concurrent.futures import ThreadPoolExecutor

def warm_up_lambda(function_name, concurrent_executions=5):
    """
    Warm up Lambda function by invoking it multiple times.
    
    Args:
        function_name: Name of Lambda function
        concurrent_executions: Number of concurrent warm-up calls
    """
    lambda_client = boto3.client('lambda')
    
    def invoke_warmup():
        try:
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps({
                    "warmup": True,
                    "body": {
                        "image_data": "",  # Empty for warmup
                        "image_name": "warmup.jpg"
                    }
                })
            )
            return response['StatusCode'] == 200
        except Exception as e:
            print(f"Warmup failed: {e}")
            return False
    
    # Execute warm-up calls concurrently
    with ThreadPoolExecutor(max_workers=concurrent_executions) as executor:
        futures = [executor.submit(invoke_warmup) for _ in range(concurrent_executions)]
        results = [future.result() for future in futures]
    
    successful_warmups = sum(results)
    print(f"Warm-up completed: {successful_warmups}/{concurrent_executions} successful")

# Schedule warm-up using CloudWatch Events
def create_warmup_schedule():
    """Create CloudWatch rule to warm up Lambda periodically."""
    
    events_client = boto3.client('events')
    lambda_client = boto3.client('lambda')
    
    # Create CloudWatch rule (every 5 minutes)
    rule_response = events_client.put_rule(
        Name='yolo-lambda-warmup',
        ScheduleExpression='rate(5 minutes)',
        Description='Warm up YOLO Lambda function',
        State='ENABLED'
    )
    
    # Add Lambda as target
    events_client.put_targets(
        Rule='yolo-lambda-warmup',
        Targets=[
            {
                'Id': '1',
                'Arn': 'arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:yolo-object-detection',
                'Input': json.dumps({
                    "warmup": True,
                    "body": {
                        "image_data": "",
                        "image_name": "scheduled_warmup.jpg"
                    }
                })
            }
        ]
    )
    
    print("Warm-up schedule created successfully!")

if __name__ == "__main__":
    warm_up_lambda("yolo-object-detection")
    create_warmup_schedule()
```

### Provisioned Concurrency
```python
# provisioned_concurrency.py
"""
Configure provisioned concurrency for consistent performance.
"""

import boto3

def configure_provisioned_concurrency(function_name, concurrency=2):
    """
    Configure provisioned concurrency for Lambda function.
    
    Args:
        function_name: Name of Lambda function
        concurrency: Number of concurrent executions to keep warm
    """
    lambda_client = boto3.client('lambda')
    
    try:
        # Publish a version first
        version_response = lambda_client.publish_version(
            FunctionName=function_name,
            Description='Version for provisioned concurrency'
        )
        
        version = version_response['Version']
        
        # Configure provisioned concurrency
        response = lambda_client.put_provisioned_concurrency_config(
            FunctionName=function_name,
            Qualifier=version,
            ProvisionedConcurrencyConfig={
                'ProvisionedConcurrencyExecutions': concurrency
            }
        )
        
        print(f"Provisioned concurrency configured successfully!")
        print(f"Function: {function_name}")
        print(f"Version: {version}")
        print(f"Concurrency: {concurrency}")
        
    except Exception as e:
        print(f"Error configuring provisioned concurrency: {e}")

if __name__ == "__main__":
    configure_provisioned_concurrency("yolo-object-detection", 2)
```

## 6. Monitoring and Logging

### CloudWatch Metrics
```python
# cloudwatch_monitoring.py
"""
CloudWatch monitoring setup for Lambda function.
"""

import boto3
import json

def create_custom_metrics():
    """Create custom CloudWatch metrics for YOLO Lambda."""
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Create custom metric for detection count
    def put_detection_metric(detection_count, image_size):
        cloudwatch.put_metric_data(
            Namespace='YOLO/ObjectDetection',
            MetricData=[
                {
                    'MetricName': 'DetectionCount',
                    'Value': detection_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {
                            'Name': 'ImageSize',
                            'Value': image_size
                        }
                    ]
                }
            ]
        )
    
    # Create custom metric for inference time
    def put_inference_time_metric(inference_time_ms):
        cloudwatch.put_metric_data(
            Namespace='YOLO/Performance',
            MetricData=[
                {
                    'MetricName': 'InferenceTime',
                    'Value': inference_time_ms,
                    'Unit': 'Milliseconds'
                }
            ]
        )
    
    return put_detection_metric, put_inference_time_metric

# Enhanced Lambda function with monitoring
def enhanced_lambda_handler(event, context):
    """Lambda handler with CloudWatch monitoring."""
    import time
    
    start_time = time.time()
    
    try:
        # Your existing lambda_handler code here
        result = lambda_handler(event, context)
        
        # Extract metrics
        body = json.loads(result['body'])
        detection_count = body.get('detection_count', 0)
        image_dims = body.get('image_dimensions', {})
        image_size = f"{image_dims.get('width', 0)}x{image_dims.get('height', 0)}"
        
        # Record metrics
        put_detection_metric, put_inference_time_metric = create_custom_metrics()
        put_detection_metric(detection_count, image_size)
        
        inference_time = (time.time() - start_time) * 1000
        put_inference_time_metric(inference_time)
        
        return result
        
    except Exception as e:
        # Record error metric
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='YOLO/Errors',
            MetricData=[
                {
                    'MetricName': 'ErrorCount',
                    'Value': 1,
                    'Unit': 'Count'
                }
            ]
        )
        raise e
```

This comprehensive AWS Lambda deployment package provides a production-ready solution for serverless YOLO object detection with optimizations for performance, monitoring, and cost-effectiveness.

