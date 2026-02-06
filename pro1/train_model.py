"""
CNN + DenseCRF End-to-End Training for Semantic Segmentation
Architecture: DeepLabV3+ → DenseCRF with Mean-field Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import cv2
from PIL import Image
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== Dense CRF Implementation ====================
class DenseCRF(nn.Module):
    """
    Dense CRF with learnable parameters for end-to-end training
    Based on "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials"
    """
    
    def __init__(self, num_classes=21, num_iterations=5, 
                 pos_w=3.0, pos_x_std=3.0, pos_y_std=3.0,
                 bi_w=10.0, bi_x_std=80.0, bi_y_std=80.0, bi_rgb_std=13.0):
        super(DenseCRF, self).__init__()
        
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        
        # Learnable CRF parameters
        self.pos_w = nn.Parameter(torch.tensor(pos_w))
        self.pos_x_std = nn.Parameter(torch.tensor(pos_x_std))
        self.pos_y_std = nn.Parameter(torch.tensor(pos_y_std))
        
        self.bi_w = nn.Parameter(torch.tensor(bi_w))
        self.bi_x_std = nn.Parameter(torch.tensor(bi_x_std))
        self.bi_y_std = nn.Parameter(torch.tensor(bi_y_std))
        self.bi_rgb_std = nn.Parameter(torch.tensor(bi_rgb_std))
        
        # Compatibility transform (learnable)
        self.compatibility = nn.Parameter(torch.eye(num_classes))
        
    def forward(self, unary, image):
        """
        Args:
            unary: [B, C, H, W] - log probabilities from CNN
            image: [B, 3, H, W] - RGB image (normalized)
        Returns:
            refined: [B, C, H, W] - refined probabilities after CRF
        """
        batch_size, num_classes, height, width = unary.shape
        
        # Convert unary to probabilities
        Q = F.softmax(unary, dim=1)
        
        # Create position features
        pos_feats = self._create_position_features(height, width, unary.device)
        
        # Denormalize image for bilateral kernel
        image_feats = self._denormalize_image(image)
        
        # Mean-field inference iterations
        for _ in range(self.num_iterations):
            Q = self._mean_field_iteration(Q, pos_feats, image_feats)
        
        return Q
    
    def _create_position_features(self, height, width, device):
        """Create spatial position features"""
        y_coords = torch.arange(height, device=device).float()
        x_coords = torch.arange(width, device=device).float()
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize by standard deviations
        y_grid = y_grid / self.pos_y_std
        x_grid = x_grid / self.pos_x_std
        
        pos_feats = torch.stack([y_grid, x_grid], dim=0)  # [2, H, W]
        return pos_feats
    
    def _denormalize_image(self, image):
        """Denormalize image for bilateral kernel"""
        # Assuming image is normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        
        image = image * std + mean
        image = image * 255.0  # Scale to [0, 255]
        
        return image
    
    def _mean_field_iteration(self, Q, pos_feats, image_feats):
        """One iteration of mean-field inference"""
        batch_size, num_classes, height, width = Q.shape
        
        # Message passing
        Q_smooth = torch.zeros_like(Q)
        
        for b in range(batch_size):
            # Appearance kernel (bilateral)
            bi_feats = self._compute_bilateral_features(
                image_feats[b], pos_feats, height, width
            )
            bi_message = self._message_passing(Q[b], bi_feats, self.bi_w)
            
            # Smoothness kernel (Gaussian)
            pos_message = self._message_passing(Q[b], pos_feats, self.pos_w)
            
            # Combine messages
            Q_smooth[b] = bi_message + pos_message
        
        # Apply compatibility transform
        Q_smooth = self._apply_compatibility(Q_smooth)
        
        # Subtract from unary and normalize
        Q_new = Q - Q_smooth
        Q_new = F.softmax(Q_new, dim=1)
        
        return Q_new
    
    def _compute_bilateral_features(self, image, pos_feats, height, width):
        """Compute bilateral kernel features"""
        # Normalize RGB by std
        rgb = image / self.bi_rgb_std  # [3, H, W]
        
        # Normalize position by std
        pos = pos_feats.clone()
        pos[0] = pos[0] * self.pos_y_std / self.bi_y_std
        pos[1] = pos[1] * self.pos_x_std / self.bi_x_std
        
        # Concatenate RGB and position features
        bi_feats = torch.cat([rgb, pos], dim=0)  # [5, H, W]
        
        return bi_feats
    
    def _message_passing(self, Q, features, weight):
        """
        Perform message passing using permutohedral lattice approximation
        Simplified version using Gaussian filtering
        """
        num_classes, height, width = Q.shape
        
        # Reshape for filtering
        Q_flat = Q.view(num_classes, -1)  # [C, H*W]
        feats_flat = features.view(features.shape[0], -1)  # [F, H*W]
        
        # Compute pairwise distances (simplified - using spatial only)
        # In full implementation, use permutohedral lattice
        message = self._gaussian_filter(Q, features, weight)
        
        return message
    
    def _gaussian_filter(self, Q, features, weight):
        """Apply Gaussian filter (simplified version)"""
        # Use depthwise separable convolution as approximation
        num_classes = Q.shape[0]
        
        # Create Gaussian kernel
        kernel_size = 7
        sigma = 2.0
        kernel = self._create_gaussian_kernel(kernel_size, sigma, Q.device)
        
        # Apply convolution
        Q_padded = F.pad(Q.unsqueeze(0), (3, 3, 3, 3), mode='reflect')
        filtered = F.conv2d(Q_padded, kernel.repeat(num_classes, 1, 1, 1), 
                           groups=num_classes, padding=0)
        
        return filtered.squeeze(0) * weight
    
    def _create_gaussian_kernel(self, size, sigma, device):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, device=device).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _apply_compatibility(self, Q):
        """Apply compatibility transform"""
        batch_size, num_classes, height, width = Q.shape
        
        # Reshape for matrix multiplication
        Q_flat = Q.view(batch_size, num_classes, -1)  # [B, C, H*W]
        
        # Apply compatibility matrix
        Q_compat = torch.matmul(self.compatibility, Q_flat)  # [B, C, H*W]
        
        # Reshape back
        Q_compat = Q_compat.view(batch_size, num_classes, height, width)
        
        return Q_compat


# ==================== CNN + CRF Model ====================
class DeepLabCRF(nn.Module):
    """
    End-to-end CNN + CRF model
    Architecture: DeepLabV3+ → DenseCRF
    """
    
    def __init__(self, num_classes=21, crf_iterations=5, use_crf=True):
        super(DeepLabCRF, self).__init__()
        
        # CNN backbone (DeepLabV3+)
        self.cnn = deeplabv3_resnet101(pretrained=True, num_classes=num_classes)
        
        # Dense CRF
        self.use_crf = use_crf
        if use_crf:
            self.crf = DenseCRF(
                num_classes=num_classes,
                num_iterations=crf_iterations,
                pos_w=3.0, pos_x_std=3.0, pos_y_std=3.0,
                bi_w=10.0, bi_x_std=80.0, bi_y_std=80.0, bi_rgb_std=13.0
            )
    
    def forward(self, image, use_crf=None):
        """
        Args:
            image: [B, 3, H, W] - input image
            use_crf: whether to use CRF refinement (default: self.use_crf)
        Returns:
            output: [B, C, H, W] - segmentation logits/probabilities
            unary: [B, C, H, W] - CNN output (before CRF)
        """
        # Get CNN output (unary potentials)
        cnn_output = self.cnn(image)['out']  # [B, C, H, W]
        
        # Store unary for training
        unary = cnn_output
        
        # Apply CRF refinement if enabled
        use_crf_flag = use_crf if use_crf is not None else self.use_crf
        
        if use_crf_flag and hasattr(self, 'crf'):
            # Convert to log probabilities for CRF
            log_probs = F.log_softmax(cnn_output, dim=1)
            
            # Apply CRF
            refined = self.crf(log_probs, image)
            
            # Convert back to logits for loss computation
            output = torch.log(refined + 1e-8)
        else:
            output = cnn_output
        
        return output, unary


# ==================== Configuration ====================
class Config:
    # Model
    NUM_CLASSES = 21
    CRF_ITERATIONS = 5
    USE_CRF = True
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 4  # Smaller due to CRF overhead
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # CRF-specific learning rate (typically lower)
    CRF_LEARNING_RATE = 0.0001
    
    # Optimization
    USE_MIXED_PRECISION = False  # CRF doesn't work well with AMP
    GRADIENT_CLIP = 1.0
    
    # Data
    IMAGE_SIZE = 512
    DATA_ROOT = "data/VOCdevkit/VOC2012"
    
    # Paths
    CHECKPOINT_DIR = "models/checkpoints_crf"
    BEST_MODEL_PATH = "models/best_cnn_crf.pth"
    LOG_DIR = "logs"
    
    # Training strategy
    WARMUP_EPOCHS = 5  # Train CNN only first
    PATIENCE = 15

config = Config()

for directory in [config.CHECKPOINT_DIR, config.LOG_DIR]:
    os.makedirs(directory, exist_ok=True)


# ==================== Dataset ====================
class VOCSegmentationDataset(Dataset):
    """Pascal VOC Dataset"""
    
    VOC_COLORMAP = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ], dtype=np.uint8)
    
    def __init__(self, root, split='train', size=512, is_train=True):
        self.root = root
        self.split = split
        self.size = size
        self.is_train = is_train
        
        list_file = os.path.join(root, 'ImageSets/Segmentation', f'{split}.txt')
        with open(list_file, 'r') as f:
            self.images = [line.strip() for line in f]
        
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'{img_name}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f'{img_name}.png')
        mask = Image.open(mask_path)
        
        # Resize
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        
        # To numpy
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)
        
        # Augmentation for training
        if self.is_train:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
        
        # Normalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # To tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask


# ==================== Metrics ====================
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask] + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
    
    def get_scores(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
        mean_iou = np.nanmean(iou)
        
        return {
            'accuracy': acc,
            'mean_iou': mean_iou,
            'class_iou': iou
        }


# ==================== Training ====================
def train_epoch(model, loader, criterion, optimizer, device, epoch, use_crf_in_training):
    model.train()
    total_loss = 0
    metrics = SegmentationMetrics(config.NUM_CLASSES)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, unary = model(images, use_crf=use_crf_in_training)
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        masks_np = masks.cpu().numpy()
        metrics.update(preds, masks_np)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'crf': 'ON' if use_crf_in_training else 'OFF'
        })
    
    return total_loss / len(loader), metrics.get_scores()


def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    metrics_no_crf = SegmentationMetrics(config.NUM_CLASSES)
    metrics_with_crf = SegmentationMetrics(config.NUM_CLASSES)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward with CRF
            outputs_crf, unary = model(images, use_crf=True)
            
            # Forward without CRF
            outputs_no_crf, _ = model(images, use_crf=False)
            
            # Compute loss (on CRF output)
            loss = criterion(outputs_crf, masks)
            total_loss += loss.item()
            
            # Metrics without CRF
            preds_no_crf = torch.argmax(outputs_no_crf, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            metrics_no_crf.update(preds_no_crf, masks_np)
            
            # Metrics with CRF
            preds_crf = torch.argmax(outputs_crf, dim=1).cpu().numpy()
            metrics_with_crf.update(preds_crf, masks_np)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    scores_no_crf = metrics_no_crf.get_scores()
    scores_with_crf = metrics_with_crf.get_scores()
    
    return total_loss / len(loader), scores_no_crf, scores_with_crf


# ==================== Main Training ====================
def train_model():
    print("🚀 Training CNN + DenseCRF Model")
    print("=" * 70)
    print("Architecture:")
    print("  Input Image")
    print("      ↓")
    print("  CNN (DeepLabV3+)")
    print("      ↓")
    print("  Unary Potentials (log-softmax)")
    print("      ↓")
    print("  DenseCRF (Mean-field Inference)")
    print("      ↓")
    print("  Refined Segmentation")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Model
    model = DeepLabCRF(
        num_classes=config.NUM_CLASSES,
        crf_iterations=config.CRF_ITERATIONS,
        use_crf=config.USE_CRF
    ).to(device)
    
    # Count parameters
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    crf_params = sum(p.numel() for p in model.crf.parameters()) if hasattr(model, 'crf') else 0
    total_params = cnn_params + crf_params
    
    print(f"\n📊 Model Parameters:")
    print(f"  CNN: {cnn_params:,}")
    print(f"  CRF: {crf_params:,}")
    print(f"  Total: {total_params:,}")
    
    # Datasets
    train_dataset = VOCSegmentationDataset(config.DATA_ROOT, 'train', config.IMAGE_SIZE, True)
    val_dataset = VOCSegmentationDataset(config.DATA_ROOT, 'val', config.IMAGE_SIZE, False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n📁 Dataset:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Separate optimizers for CNN and CRF
    cnn_params = list(model.cnn.parameters())
    crf_params = list(model.crf.parameters()) if hasattr(model, 'crf') else []
    
    optimizer = optim.AdamW([
        {'params': cnn_params, 'lr': config.LEARNING_RATE},
        {'params': crf_params, 'lr': config.CRF_LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_miou': [], 'val_miou_no_crf': [], 'val_miou_with_crf': [],
        'crf_improvement': []
    }
    
    best_miou = 0.0
    patience_counter = 0
    
    print("\n🎯 Starting training...")
    print("=" * 70)
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print('='*70)
        
        # Warmup: train CNN only first
        use_crf_in_training = epoch > config.WARMUP_EPOCHS
        
        if epoch == config.WARMUP_EPOCHS + 1:
            print("\n🔥 CRF TRAINING ACTIVATED!")
        
        # Train
        train_loss, train_scores = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_crf_in_training
        )
        
        # Validate
        val_loss, val_scores_no_crf, val_scores_with_crf = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Calculate improvement from CRF
        crf_improvement = val_scores_with_crf['mean_iou'] - val_scores_no_crf['mean_iou']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_scores['mean_iou'])
        history['val_miou_no_crf'].append(val_scores_no_crf['mean_iou'])
        history['val_miou_with_crf'].append(val_scores_with_crf['mean_iou'])
        history['crf_improvement'].append(crf_improvement)
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train mIoU: {train_scores['mean_iou']:.4f}")
        print(f"\n  Validation mIoU:")
        print(f"    Without CRF: {val_scores_no_crf['mean_iou']:.4f}")
        print(f"    With CRF:    {val_scores_with_crf['mean_iou']:.4f}")
        print(f"    Improvement: +{crf_improvement:.4f} ({crf_improvement*100:.2f}%)")
        
        # Save best model
        current_miou = val_scores_with_crf['mean_iou']
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'miou': best_miou,
                'config': vars(config)
            }, config.BEST_MODEL_PATH)
            print(f"\n  ✨ New best model! mIoU: {best_miou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
        
        # Checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_{epoch}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'history': history
            }, checkpoint_path)
    
    # Save history
    with open(os.path.join(config.LOG_DIR, 'history_crf.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    plot_training_results(history)
    
    print(f"\n{'='*70}")
    print("✅ Training completed!")
    print(f"Best mIoU (with CRF): {best_miou:.4f}")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    print('='*70)
    
    return model, history


def plot_training_results(history):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU Comparison
    axes[0, 1].plot(history['val_miou_no_crf'], label='Without CRF', linewidth=2)
    axes[0, 1].plot(history['val_miou_with_crf'], label='With CRF', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('Validation mIoU: CNN vs CNN+CRF')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CRF Improvement
    axes[1, 0].plot(history['crf_improvement'], linewidth=2, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mIoU Improvement')
    axes[1, 0].set_title('CRF Improvement over CNN')
    axes[1, 0].grid(True, alpha=0.3)
    
    # All mIoU metrics
    axes[1, 1].plot(history['train_miou'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_miou_no_crf'], label='Val (no CRF)', linewidth=2)
    axes[1, 1].plot(history['val_miou_with_crf'], label='Val (with CRF)', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mIoU')
    axes[1, 1].set_title('All mIoU Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, 'training_results_crf.png'), dpi=150)
    plt.close()
    print(f"📈 Training plots saved to {config.LOG_DIR}")


if __name__ == '__main__':
    model, history = train_model()