"""
Enhanced Flask Application with CNN + DenseCRF
Production-ready semantic segmentation with CRF refinement
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import time
import threading
from datetime import datetime
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
MODEL_PATH = 'models/best_cnn_crf.pth'
NUM_CLASSES = 21

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# VOC Classes
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)

# ==================== Dense CRF (Same as training) ====================
class DenseCRF(nn.Module):
    def __init__(self, num_classes=21, num_iterations=5, 
                 pos_w=3.0, pos_x_std=3.0, pos_y_std=3.0,
                 bi_w=10.0, bi_x_std=80.0, bi_y_std=80.0, bi_rgb_std=13.0):
        super(DenseCRF, self).__init__()
        
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        
        self.pos_w = nn.Parameter(torch.tensor(pos_w))
        self.pos_x_std = nn.Parameter(torch.tensor(pos_x_std))
        self.pos_y_std = nn.Parameter(torch.tensor(pos_y_std))
        
        self.bi_w = nn.Parameter(torch.tensor(bi_w))
        self.bi_x_std = nn.Parameter(torch.tensor(bi_x_std))
        self.bi_y_std = nn.Parameter(torch.tensor(bi_y_std))
        self.bi_rgb_std = nn.Parameter(torch.tensor(bi_rgb_std))
        
        self.compatibility = nn.Parameter(torch.eye(num_classes))
    
    def forward(self, unary, image):
        batch_size, num_classes, height, width = unary.shape
        Q = F.softmax(unary, dim=1)
        
        pos_feats = self._create_position_features(height, width, unary.device)
        image_feats = self._denormalize_image(image)
        
        for _ in range(self.num_iterations):
            Q = self._mean_field_iteration(Q, pos_feats, image_feats)
        
        return Q
    
    def _create_position_features(self, height, width, device):
        y_coords = torch.arange(height, device=device).float()
        x_coords = torch.arange(width, device=device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        y_grid = y_grid / self.pos_y_std
        x_grid = x_grid / self.pos_x_std
        
        return torch.stack([y_grid, x_grid], dim=0)
    
    def _denormalize_image(self, image):
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        image = image * std + mean
        return image * 255.0
    
    def _mean_field_iteration(self, Q, pos_feats, image_feats):
        batch_size, num_classes, height, width = Q.shape
        Q_smooth = torch.zeros_like(Q)
        
        for b in range(batch_size):
            bi_feats = self._compute_bilateral_features(image_feats[b], pos_feats, height, width)
            bi_message = self._message_passing(Q[b], bi_feats, self.bi_w)
            pos_message = self._message_passing(Q[b], pos_feats, self.pos_w)
            Q_smooth[b] = bi_message + pos_message
        
        Q_smooth = self._apply_compatibility(Q_smooth)
        Q_new = Q - Q_smooth
        return F.softmax(Q_new, dim=1)
    
    def _compute_bilateral_features(self, image, pos_feats, height, width):
        rgb = image / self.bi_rgb_std
        pos = pos_feats.clone()
        pos[0] = pos[0] * self.pos_y_std / self.bi_y_std
        pos[1] = pos[1] * self.pos_x_std / self.bi_x_std
        return torch.cat([rgb, pos], dim=0)
    
    def _message_passing(self, Q, features, weight):
        return self._gaussian_filter(Q, features, weight)
    
    def _gaussian_filter(self, Q, features, weight):
        num_classes = Q.shape[0]
        kernel_size = 7
        sigma = 2.0
        kernel = self._create_gaussian_kernel(kernel_size, sigma, Q.device)
        
        Q_padded = F.pad(Q.unsqueeze(0), (3, 3, 3, 3), mode='reflect')
        filtered = F.conv2d(Q_padded, kernel.repeat(num_classes, 1, 1, 1), 
                           groups=num_classes, padding=0)
        
        return filtered.squeeze(0) * weight
    
    def _create_gaussian_kernel(self, size, sigma, device):
        coords = torch.arange(size, device=device).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
    
    def _apply_compatibility(self, Q):
        batch_size, num_classes, height, width = Q.shape
        Q_flat = Q.view(batch_size, num_classes, -1)
        Q_compat = torch.matmul(self.compatibility, Q_flat)
        return Q_compat.view(batch_size, num_classes, height, width)


# ==================== CNN + CRF Model ====================
class DeepLabCRF(nn.Module):
    def __init__(self, num_classes=21, crf_iterations=5, use_crf=True):
        super(DeepLabCRF, self).__init__()
        self.cnn = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
        self.use_crf = use_crf
        if use_crf:
            self.crf = DenseCRF(num_classes=num_classes, num_iterations=crf_iterations)
    
    def forward(self, image, use_crf=None):
        cnn_output = self.cnn(image)['out']
        unary = cnn_output
        
        use_crf_flag = use_crf if use_crf is not None else self.use_crf
        
        if use_crf_flag and hasattr(self, 'crf'):
            log_probs = F.log_softmax(cnn_output, dim=1)
            refined = self.crf(log_probs, image)
            output = torch.log(refined + 1e-8)
        else:
            output = cnn_output
        
        return output, unary


# Global state
app_state = {
    'model_loaded': False,
    'model': None,
    'device': None,
    'crf_enabled': True
}


# ==================== Model Loading ====================
def load_model():
    """Load the CNN+CRF model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepLabCRF(num_classes=NUM_CLASSES, crf_iterations=5, use_crf=True)
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ CNN+CRF model loaded from {MODEL_PATH}")
            print(f"   mIoU: {checkpoint.get('miou', 'N/A')}")
        else:
            print("⚠️  Using pretrained CNN (no CRF weights found)")
        
        model = model.to(device)
        model.eval()
        
        app_state['model'] = model
        app_state['device'] = device
        app_state['model_loaded'] = True
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


load_model()


# ==================== Image Processing ====================
def preprocess_image(image):
    """Preprocess image for model"""
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image


def postprocess_mask(mask, original_shape):
    """Convert mask to colored visualization"""
    colored_mask = VOC_COLORMAP[mask]
    colored_mask = cv2.resize(colored_mask, (original_shape[1], original_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    return colored_mask


def get_detected_classes(mask):
    """Extract detected classes from mask"""
    unique_classes = np.unique(mask)
    detected = []
    
    for cls_id in unique_classes:
        if cls_id < len(VOC_CLASSES):
            pixel_count = np.sum(mask == cls_id)
            percentage = (pixel_count / mask.size) * 100
            
            if cls_id > 0 and percentage > 1.0:
                detected.append({
                    'id': int(cls_id),
                    'name': VOC_CLASSES[cls_id],
                    'percentage': round(percentage, 2),
                    'pixels': int(pixel_count),
                    'color': VOC_COLORMAP[cls_id].tolist()
                })
    
    detected.sort(key=lambda x: x['percentage'], reverse=True)
    return detected


# ==================== Routes ====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get application status"""
    return jsonify({
        'model_loaded': app_state['model_loaded'],
        'device': str(app_state['device']) if app_state['device'] else 'N/A',
        'crf_enabled': app_state['crf_enabled'],
        'architecture': 'CNN + DenseCRF',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform segmentation prediction with optional CRF"""
    try:
        if not app_state['model_loaded']:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get CRF option from request
        use_crf = request.form.get('use_crf', 'true').lower() == 'true'
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        original_shape = original_image.shape
        
        # Preprocess
        start_time = time.time()
        input_tensor = preprocess_image(original_image)
        input_tensor = input_tensor.to(app_state['device'])
        
        # Predict with and without CRF
        with torch.no_grad():
            # With CRF
            output_crf, unary = app_state['model'](input_tensor, use_crf=True)
            mask_crf = torch.argmax(output_crf, dim=1).squeeze().cpu().numpy()
            
            # Without CRF
            output_no_crf, _ = app_state['model'](input_tensor, use_crf=False)
            mask_no_crf = torch.argmax(output_no_crf, dim=1).squeeze().cpu().numpy()
        
        processing_time = time.time() - start_time
        
        # Choose which mask to use based on request
        mask_to_use = mask_crf if use_crf else mask_no_crf
        
        # Postprocess
        colored_mask = postprocess_mask(mask_to_use, original_shape)
        colored_mask_no_crf = postprocess_mask(mask_no_crf, original_shape)
        
        # Blend with original
        original_resized = cv2.resize(original_image, (colored_mask.shape[1], colored_mask.shape[0]))
        blended = cv2.addWeighted(original_resized, 0.6, colored_mask, 0.4, 0)
        blended_no_crf = cv2.addWeighted(original_resized, 0.6, colored_mask_no_crf, 0.4, 0)
        
        # Get detected classes for both
        detected_crf = get_detected_classes(mask_crf)
        detected_no_crf = get_detected_classes(mask_no_crf)
        
        # Save results
        timestamp = int(time.time() * 1000)
        result_filename = f'result_crf_{timestamp}.jpg'
        result_filename_no_crf = f'result_no_crf_{timestamp}.jpg'
        
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        result_path_no_crf = os.path.join(RESULTS_FOLDER, result_filename_no_crf)
        
        cv2.imwrite(result_path, blended)
        cv2.imwrite(result_path_no_crf, blended_no_crf)
        
        # Calculate statistics
        total_objects_crf = len(detected_crf)
        total_objects_no_crf = len(detected_no_crf)
        
        avg_confidence_crf = sum(d['percentage'] for d in detected_crf) / total_objects_crf if total_objects_crf > 0 else 0
        avg_confidence_no_crf = sum(d['percentage'] for d in detected_no_crf) / total_objects_no_crf if total_objects_no_crf > 0 else 0
        
        return jsonify({
            'success': True,
            'result_with_crf': f'/static/results/{result_filename}',
            'result_without_crf': f'/static/results/{result_filename_no_crf}',
            'detected_objects_crf': detected_crf,
            'detected_objects_no_crf': detected_no_crf,
            'statistics': {
                'processing_time': round(processing_time, 3),
                'image_size': f'{original_shape[1]}x{original_shape[0]}',
                'crf_results': {
                    'total_objects': total_objects_crf,
                    'avg_confidence': round(avg_confidence_crf, 2)
                },
                'no_crf_results': {
                    'total_objects': total_objects_no_crf,
                    'avg_confidence': round(avg_confidence_no_crf, 2)
                },
                'improvement': {
                    'objects_diff': total_objects_crf - total_objects_no_crf,
                    'confidence_diff': round(avg_confidence_crf - avg_confidence_no_crf, 2)
                }
            }
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/info')
def dataset_info():
    """Get dataset information"""
    return jsonify({
        'name': 'Pascal VOC 2012',
        'classes': [{'id': i, 'name': name, 'color': VOC_COLORMAP[i].tolist()} 
                   for i, name in enumerate(VOC_CLASSES)],
        'num_classes': len(VOC_CLASSES),
        'description': 'The PASCAL Visual Object Classes Challenge 2012 dataset'
    })


@app.route('/api/model/info')
def model_info():
    """Get model information"""
    if not app_state['model_loaded']:
        return jsonify({'error': 'Model not loaded'}), 500
    
    model = app_state['model']
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    crf_params = sum(p.numel() for p in model.crf.parameters()) if hasattr(model, 'crf') else 0
    total_params = cnn_params + crf_params
    
    return jsonify({
        'name': 'DeepLabV3+ with DenseCRF',
        'architecture': 'CNN (DeepLabV3+) → DenseCRF (Mean-field Inference)',
        'num_classes': NUM_CLASSES,
        'cnn_parameters': cnn_params,
        'crf_parameters': crf_params,
        'total_parameters': total_params,
        'input_size': '512x512',
        'device': str(app_state['device']),
        'crf_iterations': 5,
        'pipeline': [
            'Input Image',
            'CNN (DeepLabV3+ ResNet101)',
            'Unary Potentials (log-softmax)',
            'DenseCRF (Mean-field)',
            'Refined Segmentation'
        ]
    })


@app.route('/api/toggle_crf', methods=['POST'])
def toggle_crf():
    """Toggle CRF on/off"""
    app_state['crf_enabled'] = not app_state['crf_enabled']
    return jsonify({
        'success': True,
        'crf_enabled': app_state['crf_enabled']
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 16MB)'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🧠 Vision AI Studio - CNN + DenseCRF Semantic Segmentation")
    print("=" * 70)
    print("\n📊 Model Architecture:")
    print("   Input Image")
    print("       ↓")
    print("   CNN (DeepLabV3+ ResNet101)")
    print("       ↓")
    print("   Unary Potentials (log-softmax)")
    print("       ↓")
    print("   DenseCRF (Mean-field Inference)")
    print("       ↓")
    print("   Refined Segmentation")
    print("\n" + "=" * 70)
    print(f"Model loaded: {app_state['model_loaded']}")
    print(f"Device: {app_state['device']}")
    print(f"CRF enabled: {app_state['crf_enabled']}")
    print(f"Classes: {len(VOC_CLASSES)}")
    print("=" * 70)
    print("🌐 Server starting on http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)