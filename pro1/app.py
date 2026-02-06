"""
Simple Flask App for Semantic Segmentation
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import time
import json
from datetime import datetime
import numpy as np
import cv2
import random

app = Flask(__name__)

# Global state
training_state = {
    "is_training": False,
    "progress": 0,
    "epoch": 0,
    "total_epochs": 10,
    "loss": 0,
    "accuracy": 0,
    "emotion": "😊",
    "message": "Ready to train"
}

model_state = {
    "loaded": False,
    "accuracy": 0
}

# Directories
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# VOC Classes with emojis
VOC_CLASSES = [
    {'name': 'background', 'emoji': '⬛'},
    {'name': 'aeroplane', 'emoji': '✈️'},
    {'name': 'bicycle', 'emoji': '🚲'},
    {'name': 'bird', 'emoji': '🐦'},
    {'name': 'boat', 'emoji': '🚤'},
    {'name': 'bottle', 'emoji': '🍾'},
    {'name': 'bus', 'emoji': '🚌'},
    {'name': 'car', 'emoji': '🚗'},
    {'name': 'cat', 'emoji': '🐱'},
    {'name': 'chair', 'emoji': '🪑'},
    {'name': 'cow', 'emoji': '🐄'},
    {'name': 'diningtable', 'emoji': '🍽️'},
    {'name': 'dog', 'emoji': '🐕'},
    {'name': 'horse', 'emoji': '🐎'},
    {'name': 'motorbike', 'emoji': '🏍️'},
    {'name': 'person', 'emoji': '👤'},
    {'name': 'pottedplant', 'emoji': '🌿'},
    {'name': 'sheep', 'emoji': '🐑'},
    {'name': 'sofa', 'emoji': '🛋️'},
    {'name': 'train', 'emoji': '🚂'},
    {'name': 'tvmonitor', 'emoji': '📺'}
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Get current status"""
    # Update emotion based on training state
    if training_state["is_training"]:
        emotions = ["😊", "🤔", "🧠", "⚡", "🚀", "📈", "🎯"]
        training_state["emotion"] = random.choice(emotions)
        training_state["message"] = f"Training epoch {training_state['epoch']}/{training_state['total_epochs']}"
    else:
        training_state["emotion"] = "😊"
        training_state["message"] = "Ready to train"
    
    return jsonify({
        "training": training_state,
        "model": model_state,
        "timestamp": datetime.now().isoformat(),
        "system_emotion": random.choice(["😊", "🤔", "🚀", "🎯"])
    })

@app.route('/api/train/start', methods=['POST'])
def start_training():
    if training_state["is_training"]:
        return jsonify({"error": "Training already in progress"}), 400
    
    try:
        data = request.json or {}
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 4)
        
        training_state.update({
            "is_training": True,
            "progress": 0,
            "epoch": 0,
            "total_epochs": epochs,
            "loss": 2.0,
            "accuracy": 5.0,
            "emotion": "🚀",
            "message": "Starting training..."
        })
        
        # Start training thread
        thread = threading.Thread(
            target=simulate_training,
            args=(epochs,)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Training started for {epochs} epochs",
            "emotion": "🚀"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def simulate_training(epochs):
    """Simulate training progress"""
    try:
        for epoch in range(epochs):
            if not training_state["is_training"]:
                break
            
            # Simulate progress
            progress = (epoch + 1) / epochs * 100
            loss = max(0.1, 2.0 - (epoch * 0.18) + np.random.random() * 0.1)
            accuracy = min(95, 5 + (epoch * 9) + np.random.random() * 5)
            
            # Different emotions based on progress
            if progress < 30:
                emotion = "🤔"
            elif progress < 60:
                emotion = "🧠"
            elif progress < 90:
                emotion = "⚡"
            else:
                emotion = "🎯"
            
            training_state.update({
                "progress": round(progress, 1),
                "epoch": epoch + 1,
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 2),
                "emotion": emotion,
                "message": f"Training... ({progress:.1f}%)"
            })
            
            time.sleep(1.5)
        
        if training_state["is_training"]:
            training_state["is_training"] = False
            training_state["emotion"] = "✅"
            training_state["message"] = "Training completed!"
            model_state.update({
                "loaded": True,
                "accuracy": round(training_state["accuracy"], 2)
            })
            
    except Exception as e:
        training_state["is_training"] = False
        training_state["emotion"] = "😢"
        training_state["message"] = "Training failed"

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    training_state["is_training"] = False
    training_state["emotion"] = "⏹️"
    training_state["message"] = "Training stopped"
    return jsonify({
        "success": True, 
        "message": "Training stopped",
        "emotion": "⏹️"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    if 'image' not in request.files:
        return jsonify({"error": "No image selected"}), 400
    
    try:
        # Save uploaded file
        image_file = request.files['image']
        filename = f"pred_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)
        
        # Check if model is loaded
        if not model_state["loaded"]:
            # Create a mock model for demo
            model_state["loaded"] = True
            model_state["accuracy"] = 85.5
        
        # Generate result
        result_filename = f"result_{int(time.time())}_{random.randint(1000, 9999)}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        # Read and process image
        original = cv2.imread(filepath)
        if original is None:
            return jsonify({"error": "Could not read image"}), 400
        
        # Resize for display
        height, width = 400, 500
        original = cv2.resize(original, (width, height))
        
        # Create segmentation mask
        segmented = create_segmentation_mask(original)
        
        # Blend with original
        alpha = 0.6
        blended = cv2.addWeighted(original, alpha, segmented, 1-alpha, 0)
        
        # Add labels
        blended = add_segmentation_labels(blended)
        
        cv2.imwrite(result_path, blended)
        
        # Generate random detected objects
        detected_classes = random.sample(VOC_CLASSES[1:], random.randint(2, 5))
        detected = [f"{cls['emoji']} {cls['name']}" for cls in detected_classes]
        
        # Confidence based on model accuracy
        confidence = min(0.95, model_state["accuracy"] / 100 * 0.9 + np.random.random() * 0.1)
        
        # Emotion based on confidence
        if confidence > 0.9:
            emotion = "🎯"
            message = "Excellent segmentation!"
        elif confidence > 0.7:
            emotion = "😊"
            message = "Good segmentation results!"
        else:
            emotion = "🤔"
            message = "Decent segmentation results"
        
        return jsonify({
            "success": True,
            "message": message,
            "emotion": emotion,
            "original": f"/static/uploads/{filename}",
            "result": f"/static/results/{result_filename}",
            "detected": detected,
            "confidence": round(confidence, 2),
            "time_taken": round(np.random.uniform(0.5, 2.5), 1)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_segmentation_mask(image):
    """Create a mock segmentation mask"""
    height, width = image.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add random colored segments
    colors = [
        (255, 100, 100),   # Red
        (100, 255, 100),   # Green
        (100, 100, 255),   # Blue
        (255, 255, 100),   # Yellow
        (255, 100, 255),   # Magenta
        (100, 255, 255),   # Cyan
        (200, 150, 100),   # Brown
        (150, 100, 200),   # Purple
    ]
    
    # Create random blobs
    for _ in range(random.randint(3, 8)):
        color = random.choice(colors)
        center_x = random.randint(50, width - 50)
        center_y = random.randint(50, height - 50)
        radius = random.randint(30, 80)
        
        # Create blob with random shape
        if random.random() > 0.5:
            cv2.circle(mask, (center_x, center_y), radius, color, -1)
        else:
            x1 = random.randint(20, width - 100)
            y1 = random.randint(20, height - 100)
            x2 = x1 + random.randint(60, 120)
            y2 = y1 + random.randint(60, 120)
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
    
    # Add some noise for realism
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    mask = cv2.add(mask, noise)
    
    # Apply Gaussian blur for smooth edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    return mask

def add_segmentation_labels(image):
    """Add labels to segmented image"""
    height, width = image.shape[:2]
    
    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Semantic Segmentation', (width//2 - 150, 30), 
                font, 0.7, (255, 255, 255), 2)
    
    # Add border
    cv2.rectangle(image, (5, 5), (width-5, height-5), (67, 97, 238), 2)
    
    return image

@app.route('/api/dataset/info')
def dataset_info():
    """Get dataset information"""
    return jsonify({
        "name": "Pascal VOC 2012",
        "total_images": "17,125",
        "training_set": 1464,
        "validation_set": 1449,
        "classes": VOC_CLASSES,
        "class_count": len(VOC_CLASSES)
    })

@app.route('/api/logs')
def get_logs():
    """Get recent activity logs"""
    logs = [
        {"time": "10:30", "message": "System initialized", "emoji": "🚀"},
        {"time": "10:31", "message": "Dataset loaded successfully", "emoji": "📊"},
        {"time": "10:32", "message": "Model ready for training", "emoji": "🤖"},
    ]
    return jsonify(logs)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("🎨 Starting AI Vision Studio...")
    print("🌐 Open http://localhost:5000 in your browser")
    print("✨ Features:")
    print("   - Semantic Segmentation Training")
    print("   - Real-time Progress Animations")
    print("   - Image Prediction with Visualizations")
    print("   - Interactive Dataset Explorer")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)