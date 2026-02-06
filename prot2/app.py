"""
Flask web application for semantic segmentation
"""
import os
import sys
import threading
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

from config import Config
from utils import prepare_dataset_info, predict_single_image, save_prediction_results
from train_model import train_model_background, SegmentationTrainer
from setup_data import DatasetSetup

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Global training state
training_state = {
    "is_training": False,
    "status": "idle",
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": Config.NUM_EPOCHS,
    "current_loss": 0,
    "current_iou": 0,
    "best_iou": 0,
    "logs": [],
    "start_time": None,
    "training_thread": None
}

# Global model cache
model_cache = {
    "model": None,
    "device": None,
    "last_loaded": None
}

def log_message(message: str, level: str = "INFO"):
    """Add message to training logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    
    training_state["logs"].append(log_entry)
    
    # Keep only last 100 logs
    if len(training_state["logs"]) > 100:
        training_state["logs"] = training_state["logs"][-100:]
    
    print(log_entry)

def get_device():
    """Get available device"""
    if Config.USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_model(model_path: str = None):
    """Load trained model"""
    try:
        from utils import SimpleSegmentationModel
        
        # Determine which model to load
        if model_path is None:
            # Try to load latest model
            model_path = Path(Config.MODEL_FOLDER) / "latest_model.pth"
            if not model_path.exists():
                model_path = Path(Config.MODEL_FOLDER) / "final_model.pth"
                if not model_path.exists():
                    return None, "No trained model found"
        
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"
        
        # Initialize model
        device = get_device()
        model = SimpleSegmentationModel(num_classes=Config.NUM_CLASSES)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Update cache
        model_cache["model"] = model
        model_cache["device"] = device
        model_cache["last_loaded"] = time.time()
        
        log_message(f"Model loaded from {model_path}")
        return model, None
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def training_callback(status: str, data: Any):
    """Callback for training updates"""
    if status == "progress":
        training_state["progress"] = data.get("progress", 0)
        training_state["current_epoch"] = data.get("epoch", 0)
        training_state["current_loss"] = data.get("loss", 0)
        training_state["current_iou"] = data.get("iou", 0)
        training_state["best_iou"] = data.get("best_iou", 0)
        
        log_message(f"Epoch {data.get('epoch', 0)}: "
                   f"Loss={data.get('loss', 0):.4f}, "
                   f"IoU={data.get('iou', 0):.4f}")
    
    elif status == "complete":
        training_state["is_training"] = False
        training_state["status"] = "completed"
        training_state["progress"] = 100
        
        log_message(f"Training completed! Best IoU: {data.get('best_iou', 0):.4f}")
        
        # Load the trained model
        load_model()
    
    elif status == "error":
        training_state["is_training"] = False
        training_state["status"] = "error"
        
        log_message(f"Training error: {data}", "ERROR")

# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    """Get dataset information"""
    info = prepare_dataset_info()
    return jsonify(info)

@app.route('/api/dataset/setup', methods=['POST'])
def setup_dataset():
    """Setup dataset"""
    try:
        dataset_path = DatasetSetup.setup()
        
        if dataset_path:
            return jsonify({
                "success": True,
                "message": f"Dataset setup complete at {dataset_path}",
                "path": str(dataset_path)
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to setup dataset"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    # Calculate elapsed time
    elapsed = None
    if training_state["start_time"]:
        elapsed = time.time() - training_state["start_time"]
    
    response = {
        "is_training": training_state["is_training"],
        "status": training_state["status"],
        "progress": training_state["progress"],
        "current_epoch": training_state["current_epoch"],
        "total_epochs": training_state["total_epochs"],
        "current_loss": training_state["current_loss"],
        "current_iou": training_state["current_iou"],
        "best_iou": training_state["best_iou"],
        "elapsed_time": elapsed,
        "logs": training_state["logs"][-20:]  # Last 20 logs
    }
    
    return jsonify(response)

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    if training_state["is_training"]:
        return jsonify({
            "success": False,
            "message": "Training already in progress"
        }), 400
    
    try:
        # Get training parameters
        data = request.get_json()
        
        epochs = data.get('epochs', Config.NUM_EPOCHS)
        batch_size = data.get('batch_size', Config.BATCH_SIZE)
        learning_rate = data.get('learning_rate', Config.LEARNING_RATE)
        
        # Reset training state
        training_state.update({
            "is_training": True,
            "status": "starting",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "current_loss": 0,
            "current_iou": 0,
            "best_iou": 0,
            "logs": [],
            "start_time": time.time()
        })
        
        log_message("Starting training...")
        log_message(f"Parameters: epochs={epochs}, "
                   f"batch_size={batch_size}, lr={learning_rate}")
        
        # Start training in background thread
        training_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        def train_thread():
            try:
                train_model_background(training_params, training_callback)
            except Exception as e:
                training_callback("error", str(e))
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        training_state["training_thread"] = thread
        
        return jsonify({
            "success": True,
            "message": "Training started successfully"
        })
        
    except Exception as e:
        training_state["is_training"] = False
        training_state["status"] = "error"
        
        return jsonify({
            "success": False,
            "message": f"Error starting training: {str(e)}"
        }), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop model training"""
    if not training_state["is_training"]:
        return jsonify({
            "success": False,
            "message": "No training in progress"
        }), 400
    
    # This is a simple implementation
    # In production, you'd need a way to gracefully stop the training thread
    training_state["is_training"] = False
    training_state["status"] = "stopped"
    
    log_message("Training stopped by user")
    
    return jsonify({
        "success": True,
        "message": "Training stopped"
    })

@app.route('/api/model/load', methods=['POST'])
def load_model_endpoint():
    """Load a trained model"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        model, error = load_model(model_path)
        
        if error:
            return jsonify({
                "success": False,
                "message": error
            }), 400
        
        return jsonify({
            "success": True,
            "message": "Model loaded successfully"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error loading model: {str(e)}"
        }), 500

@app.route('/api/model/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        }), 400
    
    try:
        # Get uploaded file
        image_file = request.files['image']
        
        # Save uploaded file
        upload_dir = Path(Config.UPLOAD_FOLDER)
        upload_dir.mkdir(exist_ok=True)
        
        file_ext = Path(image_file.filename).suffix
        filename = f"upload_{int(time.time())}{file_ext}"
        filepath = upload_dir / filename
        
        image_file.save(filepath)
        
        # Load model if not loaded
        if model_cache["model"] is None:
            model, error = load_model()
            if error:
                return jsonify({
                    "success": False,
                    "message": error
                }), 400
        
        # Make prediction
        results = predict_single_image(
            model_cache["model"],
            filepath,
            str(model_cache["device"])
        )
        
        # Save results
        result_files = save_prediction_results(
            results,
            Path(filename).stem
        )
        
        # Prepare response
        response = {
            "success": True,
            "message": "Prediction completed",
            "original_image": result_files["original"],
            "segmentation": result_files["segmentation"],
            "blended": result_files["blended"],
            "mask": result_files["mask"],
            "detected_classes": results["detected_classes"],
            "class_ids": results["class_ids"],
            "class_names": [Config.VOC_CLASSES[i] for i in results["class_ids"] 
                          if i < len(Config.VOC_CLASSES)]
        }
        
        log_message(f"Prediction completed for {filename}")
        
        return jsonify(response)
        
    except Exception as e:
        log_message(f"Prediction error: {str(e)}", "ERROR")
        return jsonify({
            "success": False,
            "message": f"Prediction error: {str(e)}"
        }), 500

@app.route('/api/model/list', methods=['GET'])
def list_models():
    """List available trained models"""
    model_dir = Path(Config.MODEL_FOLDER)
    checkpoint_dir = Path("checkpoints")
    
    models = []
    
    # Check model directory
    if model_dir.exists():
        for model_file in model_dir.glob("*.pth"):
            models.append({
                "name": model_file.name,
                "path": str(model_file),
                "size": model_file.stat().st_size,
                "type": "model"
            })
    
    # Check checkpoint directory
    if checkpoint_dir.exists():
        for checkpoint in checkpoint_dir.glob("*.pth"):
            models.append({
                "name": checkpoint.name,
                "path": str(checkpoint),
                "size": checkpoint.stat().st_size,
                "type": "checkpoint"
            })
    
    return jsonify({
        "success": True,
        "models": sorted(models, key=lambda x: x["size"], reverse=True)
    })

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/system/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "training_active": training_state["is_training"],
        "model_loaded": model_cache["model"] is not None,
        "gpu_available": torch.cuda.is_available() if Config.USE_GPU else False,
        "dataset_ready": DatasetSetup.find_dataset() is not None
    }
    
    return jsonify(health_info)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500

# Initialize application
def init_app():
    """Initialize the application"""
    print("=" * 60)
    print("Semantic Segmentation Web Application")
    print("=" * 60)
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Try to find dataset
    dataset_path = DatasetSetup.find_dataset()
    if dataset_path:
        print(f"✅ Dataset found at: {dataset_path}")
    else:
        print("⚠️  Dataset not found. Use the web interface to download it.")
    
    # Try to load existing model
    model, error = load_model()
    if model:
        print("✅ Pre-trained model loaded")
    else:
        print(f"ℹ️  No model loaded: {error}")
    
    print(f"\n🌐 Starting web server on http://localhost:5000")
    print("   Press Ctrl+C to stop\n")

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)