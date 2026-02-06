"""
Setup script for the application
"""
import os
import sys

def setup_environment():
    """Setup the application environment"""
    print("🚀 Setting up AI Vision Studio...")
    print("=" * 50)
    
    # Create directory structure
    directories = [
        'static/uploads',
        'static/results',
        'static/css',
        'static/js',
        'models',
        'data/dummy',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check for required directories
    required_dirs = ['templates', 'static/css', 'static/js']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    print("✅ Environment setup complete!")
    print("=" * 50)
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    print("📦 Checking dependencies...")
    
    try:
        import flask
        import numpy
        import cv2
        import torch
        
        print("✅ All dependencies are installed!")
        print(f"   Flask: {flask.__version__}")
        print(f"   NumPy: {numpy.__version__}")
        print(f"   OpenCV: {cv2.__version__}")
        print(f"   PyTorch: {torch.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def create_dummy_model():
    """Create a dummy model file for demo"""
    print("🤖 Creating dummy model...")
    
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a simple model info file
    model_info = {
        "name": "Demo Segmentation Model",
        "type": "UNet",
        "classes": 21,
        "accuracy": 85.5,
        "created": "2024-01-01",
        "description": "Pretrained model for Pascal VOC 2012"
    }
    
    import json
    with open(os.path.join(models_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Dummy model created!")
    return True

def print_startup_guide():
    """Print startup guide"""
    print("🎮 STARTUP GUIDE")
    print("=" * 50)
    print("1. Start the Flask server:")
    print("   $ python app.py")
    print("")
    print("2. Open your browser and go to:")
    print("   http://localhost:5000")
    print("")
    print("3. Available features:")
    print("   • Train segmentation models")
    print("   • Upload and predict images")
    print("   • View dataset information")
    print("   • Real-time progress animations")
    print("")
    print("4. Quick test:")
    print("   • Click 'Use Sample' to load a test image")
    print("   • Click 'Start Training' to begin")
    print("   • Click 'Analyze Image' after training")
    print("=" * 50)
    print("✨ Setup complete! Ready to launch!")
    print("")

if __name__ == "__main__":
    print("🎨 AI Vision Studio Setup")
    print("=" * 50)
    
    # Run setup steps
    if setup_environment():
        if check_dependencies():
            create_dummy_model()
            print_startup_guide()
        else:
            print("❌ Please install dependencies first")
    else:
        print("❌ Setup failed")