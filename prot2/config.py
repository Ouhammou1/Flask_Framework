"""
Configuration settings for the application
"""
import os

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'static/uploads'
    RESULT_FOLDER = 'static/results'
    MODEL_FOLDER = 'models'
    DATA_FOLDER = 'data'
    
    # Dataset settings
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # Model settings
    IMAGE_SIZE = (256, 256)
    NUM_CLASSES = 21
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # Training settings
    USE_GPU = True
    SAVE_INTERVAL = 5  # Save model every N epochs
    
    # CRF settings
    USE_CRF = True
    CRF_ITERATIONS = 5
    
    # Paths
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.RESULT_FOLDER, exist_ok=True)
        os.makedirs(cls.MODEL_FOLDER, exist_ok=True)
        os.makedirs(cls.DATA_FOLDER, exist_ok=True)