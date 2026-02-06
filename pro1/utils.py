"""
Utility functions for the application
"""
import os
import json
import numpy as np
import cv2
from datetime import datetime
import random

def ensure_directories():
    """Create necessary directories"""
    directories = [
        'static/uploads',
        'static/results',
        'models',
        'data/dummy'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def create_sample_data():
    """Create sample images for demo"""
    print("🎨 Creating sample images...")
    
    sample_dir = 'static/samples'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create 3 sample images
    for i in range(3):
        # Create random image with shapes
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img.fill(240)  # Light background
        
        # Add random shapes
        for _ in range(5):
            color = (random.randint(0, 200), 
                    random.randint(0, 200), 
                    random.randint(0, 200))
            
            if random.random() > 0.5:
                # Circle
                center = (random.randint(50, 350), random.randint(50, 250))
                radius = random.randint(20, 50)
                cv2.circle(img, center, radius, color, -1)
            else:
                # Rectangle
                x1 = random.randint(20, 300)
                y1 = random.randint(20, 200)
                x2 = x1 + random.randint(50, 100)
                y2 = y1 + random.randint(50, 100)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Add text
        cv2.putText(img, f'Sample {i+1}', (150, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save
        filename = os.path.join(sample_dir, f'sample_{i+1}.jpg')
        cv2.imwrite(filename, img)
        print(f"  Created: {filename}")
    
    print("✅ Sample images created!")

def cleanup_old_files(directory, max_files=10):
    """Clean up old files in directory"""
    if not os.path.exists(directory):
        return
    
    files = os.listdir(directory)
    if len(files) > max_files:
        # Sort by creation time
        files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
        
        # Remove oldest files
        for file in files[:-max_files]:
            filepath = os.path.join(directory, file)
            os.remove(filepath)
            print(f"🗑️  Removed old file: {filepath}")

def get_sample_images():
    """Get list of sample images"""
    sample_dir = 'static/samples'
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
        return [f'/static/samples/{sample}' for sample in samples]
    
    # Default sample URLs
    return [
        'https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400&h=300&fit=crop',
        'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400&h=300&fit=crop',
        'https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=400&h=300&fit=crop'
    ]

def log_activity(message, emoji="📝"):
    """Log activity with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"{emoji} [{timestamp}] {message}"
    
    # Print to console
    print(log_entry)
    
    # Save to log file
    with open('activity.log', 'a') as f:
        f.write(log_entry + '\n')
    
    return log_entry

def get_voc_colors():
    """Get colors for VOC classes"""
    colors = [
        (0, 0, 0),          # background
        (128, 0, 0),        # aeroplane
        (0, 128, 0),        # bicycle
        (128, 128, 0),      # bird
        (0, 0, 128),        # boat
        (128, 0, 128),      # bottle
        (0, 128, 128),      # bus
        (128, 128, 128),    # car
        (64, 0, 0),         # cat
        (192, 0, 0),        # chair
        (64, 128, 0),       # cow
        (192, 128, 0),      # diningtable
        (64, 0, 128),       # dog
        (192, 0, 128),      # horse
        (64, 128, 128),     # motorbike
        (192, 128, 128),    # person
        (0, 64, 0),         # pottedplant
        (128, 64, 0),       # sheep
        (0, 192, 0),        # sofa
        (128, 192, 0),      # train
        (0, 64, 128),       # tvmonitor
    ]
    return colors

def format_file_size(bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

if __name__ == "__main__":
    # Test utilities
    ensure_directories()
    create_sample_data()
    print("✨ Utilities ready!")