
# Phase 3: YOLOv5 Model Training and Optimization

This phase covers the practical aspects of training a YOLOv5 model on your custom dataset and introduces key optimization techniques to enhance its performance.

## 1. Setting up the Environment

To begin, you need to set up your development environment. YOLOv5 is built on PyTorch, so ensure you have a compatible Python environment.

1.  **Install PyTorch:** Follow the instructions on the official PyTorch website ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) to install PyTorch, ensuring you select the correct version for your operating system and CUDA (if you have a GPU).

2.  **Clone YOLOv5 Repository:**
    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **GPU Setup (if applicable):** If you have an NVIDIA GPU, ensure your CUDA drivers are up-to-date and PyTorch is installed with CUDA support. This will significantly speed up training.

## 2. Training a Custom YOLOv5 Model

Training YOLOv5 on your custom dataset involves a few key steps:

1.  **Prepare your `custom.yaml`:** This YAML file defines the paths to your training and validation images/labels, and the number of classes. Create a file (e.g., `data/custom.yaml`) with content similar to this:

    ```yaml
    # custom.yaml
    train: ../path/to/your/dataset/images/train  # path to training images
    val: ../path/to/your/dataset/images/val    # path to validation images

    # number of classes
    nc: 3  # e.g., 3 for apple, banana, orange

    # class names
    names: ["apple", "banana", "orange"]
    ```
    *Make sure the `train` and `val` paths point to the directories containing your images.* The `nc` should match the number of classes in your dataset, and `names` should list your class names in order of their `class_id`.

2.  **Load a Pre-trained YOLOv5 Model:** Leverage transfer learning by starting with a model pre-trained on a large dataset like COCO. This helps in faster convergence and better performance, especially with smaller datasets.

    You can download pre-trained weights (e.g., `yolov5s.pt` for a small model) from the Ultralytics GitHub releases page or let the training script download it automatically.

3.  **Applying Transfer Learning Techniques:**
    *   **Freezing Backbone Layers:** For initial training, you might want to freeze the early layers of the model (the backbone) to preserve the general features learned from the large dataset. This is often done implicitly by the training process when using pre-trained weights and a smaller learning rate.
    *   **Modifying Detection Head:** The final layers (detection head) of the YOLO model need to be adapted to predict the correct number of classes for your specific task. YOLOv5 handles this automatically when you specify `nc` in your `custom.yaml`.

4.  **Train the Model:** Execute the `train.py` script from the YOLOv5 directory:

    ```bash
    python train.py --img 640 --batch 16 --epochs 100 --data data/custom.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name custom_yolov5_run
    ```
    *   `--img 640`: Input image size for training.
    *   `--batch 16`: Batch size. Adjust based on your GPU memory.
    *   `--epochs 100`: Number of training epochs.
    *   `--data data/custom.yaml`: Path to your dataset configuration file.
    *   `--cfg models/yolov5s.yaml`: Path to the model configuration (e.g., `yolov5s.yaml` for YOLOv5s architecture).
    *   `--weights yolov5s.pt`: Path to your pre-trained weights. If not found, it will be downloaded.
    *   `--name custom_yolov5_run`: A name for your training run, results will be saved in `runs/train/custom_yolov5_run`.

5.  **Monitoring Training Progress and Evaluating Metrics:** During training, YOLOv5 will output various metrics (mAP, precision, recall, loss). You can visualize these using TensorBoard:

    ```bash
    tensorboard --logdir runs/train
    ```
    Open your browser to the address provided by TensorBoard (usually `http://localhost:6006`).

## 3. Model Optimization in Practice

After training, you can further optimize your model for deployment, especially for resource-constrained environments like AWS Lambda.

*   **Pruning and Quantization:** These techniques reduce model size and improve inference speed. While YOLOv5 itself doesn't have built-in pruning/quantization tools directly in its `train.py`, you can explore external libraries or frameworks like PyTorch's `torch.quantization` or NVIDIA's TensorRT for more advanced optimization. The `Computer_Vision_with_YOLO_From_Basics_to_Deployment.txt` mentions these as key optimization techniques.

*   **Exporting the Trained Model:** For deployment, you'll typically export your `.pt` (PyTorch) model to a more deployment-friendly format like ONNX or TorchScript.

    ```bash
    python export.py --weights runs/train/custom_yolov5_run/weights/best.pt --include onnx  # Export to ONNX
    python export.py --weights runs/train/custom_yolov5_run/weights/best.pt --include torchscript  # Export to TorchScript
    ```
    These exported models are more suitable for integration into Flask APIs and AWS Lambda functions.

