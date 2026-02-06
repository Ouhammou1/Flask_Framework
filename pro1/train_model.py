"""
Advanced Semantic Segmentation Training with DeepLabV3+
Best model for Pascal VOC 2012 dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast

import os
import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Model settings
    MODEL_NAME = "DeepLabV3+_ResNet101"
    NUM_CLASSES = 21  # Pascal VOC 2012 has 20 classes + background
    BACKBONE = "resnet101"
    OUTPUT_STRIDE = 16
    
    # Training settings
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    # Dataset paths
    DATA_ROOT = "data/VOCdevkit/VOC2012"
    TRAIN_IMAGES = os.path.join(DATA_ROOT, "JPEGImages")
    TRAIN_MASKS = os.path.join(DATA_ROOT, "SegmentationClass")
    TRAIN_LIST = os.path.join(DATA_ROOT, "ImageSets/Segmentation/train.txt")
    VAL_LIST = os.path.join(DATA_ROOT, "ImageSets/Segmentation/val.txt")
    
    # Augmentation
    IMAGE_SIZE = (512, 512)
    
    # Training
    USE_MIXED_PRECISION = True
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VALUE = 1.0
    
    # Checkpoints
    CHECKPOINT_DIR = "models/checkpoints"
    BEST_MODEL_PATH = "models/best_deeplabv3_resnet101.pth"
    FINAL_MODEL_PATH = "models/final_deeplabv3_resnet101.pth"
    
    # Logging
    LOG_DIR = "logs"
    PLOT_DIR = "plots"

config = Config()

# Create directories
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.PLOT_DIR, exist_ok=True)

# DeepLabV3+ Implementation
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    def __init__(self, in_channels, out_channels=256, output_stride=16):
        super(ASPP, self).__init__()
        
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        # ASPP modules
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], 
                     dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], 
                     dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], 
                     dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Concat and project
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet backbone"""
    def __init__(self, num_classes=21, backbone='resnet101', output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone
        if backbone == 'resnet101':
            backbone_model = models.resnet101(pretrained=True)
            self.low_level_features = 256
        elif backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=True)
            self.low_level_features = 256
        else:
            raise NotImplementedError
        
        # Early layers
        self.conv1 = backbone_model.conv1
        self.bn1 = backbone_model.bn1
        self.relu = backbone_model.relu
        self.maxpool = backbone_model.maxpool
        
        # ResNet layers
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4
        
        # Adjust dilation rates for output stride
        if output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        
        # ASPP
        self.aspp = ASPP(2048, 256, output_stride)
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(self.low_level_features, 48, 1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(48)
        self.decoder_relu = nn.ReLU(inplace=True)
        
        self.decoder_conv2 = nn.Conv2d(304, 256, 3, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_relu2 = nn.ReLU(inplace=True)
        self.decoder_dropout = nn.Dropout(0.5)
        
        self.decoder_conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.decoder_bn3 = nn.BatchNorm2d(256)
        self.decoder_relu3 = nn.ReLU(inplace=True)
        self.decoder_dropout2 = nn.Dropout(0.1)
        
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # Low-level features
        low_level_feat = x
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        low_level_feat = self.decoder_conv1(low_level_feat)
        low_level_feat = self.decoder_bn1(low_level_feat)
        low_level_feat = self.decoder_relu(low_level_feat)
        
        x = nn.functional.interpolate(x, size=low_level_feat.size()[2:], 
                                     mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_dropout(x)
        
        x = self.decoder_conv3(x)
        x = self.decoder_bn3(x)
        x = self.decoder_relu3(x)
        x = self.decoder_dropout2(x)
        
        # Final classifier
        x = self.classifier(x)
        x = nn.functional.interpolate(x, size=(512, 512), 
                                     mode='bilinear', align_corners=True)
        
        return x

# Dataset Class
class PascalVOCDataset(Dataset):
    """Pascal VOC 2012 Dataset with augmentations"""
    def __init__(self, image_dir, mask_dir, list_file, transform=None, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        
        # Read image names from list file
        with open(list_file, 'r') as f:
            self.image_names = [line.strip() for line in f if line.strip()]
        
        # Pascal VOC color map (21 classes)
        self.colormap = self.create_pascal_label_colormap()
        
        print(f"Loaded {len(self.image_names)} images from {list_file}")
    
    def create_pascal_label_colormap(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]          # background
        colormap[1] = [128, 0, 0]        # aeroplane
        colormap[2] = [0, 128, 0]        # bicycle
        colormap[3] = [128, 128, 0]      # bird
        colormap[4] = [0, 0, 128]        # boat
        colormap[5] = [128, 0, 128]      # bottle
        colormap[6] = [0, 128, 128]      # bus
        colormap[7] = [128, 128, 128]    # car
        colormap[8] = [64, 0, 0]         # cat
        colormap[9] = [192, 0, 0]        # chair
        colormap[10] = [64, 128, 0]      # cow
        colormap[11] = [192, 128, 0]     # diningtable
        colormap[12] = [64, 0, 128]      # dog
        colormap[13] = [192, 0, 128]     # horse
        colormap[14] = [64, 128, 128]    # motorbike
        colormap[15] = [192, 128, 128]   # person
        colormap[16] = [0, 64, 0]        # pottedplant
        colormap[17] = [128, 64, 0]      # sheep
        colormap[18] = [0, 192, 0]       # sofa
        colormap[19] = [128, 192, 0]     # train
        colormap[20] = [0, 64, 128]      # tvmonitor
        return colormap
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f"{image_name}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply color map to mask
        mask_colored = self.colormap[mask]
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask_colored)
            image = augmented['image']
            mask_colored = augmented['mask']
            
            # Convert colored mask back to class indices
            mask = torch.zeros(mask_colored.shape[1:], dtype=torch.long)
            for i, color in enumerate(self.colormap):
                if i >= config.NUM_CLASSES:
                    break
                color_tensor = torch.tensor(color).view(3, 1, 1)
                mask_i = (mask_colored == color_tensor).all(dim=0)
                mask[mask_i] = i
        else:
            # Basic preprocessing
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])(image)
            
            mask = torch.from_numpy(mask).long()
        
        return image, mask, image_name

# Data Augmentations
def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(height=512, width=512, scale=(0.5, 2.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Metrics Calculation
class SegmentationMetrics:
    """Calculate segmentation metrics"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
    
    def update(self, pred, target):
        """Update confusion matrix"""
        pred = pred.flatten()
        target = target.flatten()
        
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
    
    def get_scores(self):
        """Returns accuracy score evaluation metrics"""
        hist = self.confusion_matrix
        
        # Overall accuracy
        acc = np.diag(hist).sum() / hist.sum()
        
        # Per class accuracy
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        
        # Mean accuracy
        acc_cls = np.nanmean(acc_cls)
        
        # Per class IoU
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        
        # Mean IoU
        mean_iu = np.nanmean(iu)
        
        # Frequency weighted IoU
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        
        return {
            "Overall_Acc": acc,
            "Mean_Acc": acc_cls,
            "FreqW_Acc": fwavacc,
            "Mean_IoU": mean_iu,
            "Class_IoU": iu
        }
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

# Training Function
def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics = SegmentationMetrics(config.NUM_CLASSES)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            if config.USE_GRADIENT_CLIPPING:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            if config.USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
            
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        metrics.update(preds, masks_np)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (pbar.n + 1):.4f}"
        })
    
    epoch_loss = total_loss / len(dataloader)
    scores = metrics.get_scores()
    
    return epoch_loss, scores

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    metrics = SegmentationMetrics(config.NUM_CLASSES)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    with torch.no_grad():
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Update metrics
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            metrics.update(preds, masks_np)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (pbar.n + 1):.4f}"
            })
    
    epoch_loss = total_loss / len(dataloader)
    scores = metrics.get_scores()
    
    return epoch_loss, scores

# Training Loop
def train_model():
    """Main training function"""
    print("🚀 Starting DeepLabV3+ Training on Pascal VOC 2012")
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DeepLabV3Plus(
        num_classes=config.NUM_CLASSES,
        backbone=config.BACKBONE,
        output_stride=config.OUTPUT_STRIDE
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Create datasets
    print("\n📁 Loading datasets...")
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = PascalVOCDataset(
        image_dir=config.TRAIN_IMAGES,
        mask_dir=config.TRAIN_MASKS,
        list_file=config.TRAIN_LIST,
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = PascalVOCDataset(
        image_dir=config.TRAIN_IMAGES,
        mask_dir=config.TRAIN_MASKS,
        list_file=config.VAL_LIST,
        transform=val_transform,
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss function (with class weighting for Pascal VOC)
    # Pascal VOC classes are imbalanced, so we weight them
    class_weights = torch.tensor([
        0.05,  # background
        1.0, 1.0, 1.0, 1.0, 1.0,  # 1-5
        1.0, 1.0, 1.0, 1.0, 1.0,  # 6-10
        1.0, 1.0, 1.0, 1.0, 1.0,  # 11-15
        1.0, 1.0, 1.0, 1.0, 1.0   # 16-20
    ]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Optimizer with different learning rates for backbone and decoder
    backbone_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},
        {'params': decoder_params, 'lr': config.LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_miou': [],
        'val_miou': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_miou = 0.0
    patience = 10
    patience_counter = 0
    
    # Start training
    print("\n🎯 Starting training...")
    print("=" * 60)
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_scores = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Validate
        val_loss, val_scores = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_scores['Mean_IoU'])
        history['val_miou'].append(val_scores['Mean_IoU'])
        history['train_acc'].append(train_scores['Overall_Acc'])
        history['val_acc'].append(val_scores['Overall_Acc'])
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train mIoU: {train_scores['Mean_IoU']:.4f} | Val mIoU: {val_scores['Mean_IoU']:.4f}")
        print(f"Train Acc: {train_scores['Overall_Acc']:.4f} | Val Acc: {val_scores['Overall_Acc']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_miou': val_scores['Mean_IoU'],
            'history': history,
        }, checkpoint_path)
        
        # Save best model
        if val_scores['Mean_IoU'] > best_miou:
            best_miou = val_scores['Mean_IoU']
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"✨ New best model saved with mIoU: {best_miou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch} epochs")
            break
        
        # Plot progress
        if epoch % 5 == 0:
            plot_training_progress(history, epoch)
    
    # Save final model
    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
    print(f"\n✅ Final model saved to {config.FINAL_MODEL_PATH}")
    
    # Save training history
    with open(os.path.join(config.LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot final results
    plot_training_progress(history, 'final')
    
    print("\n🎉 Training completed!")
    print(f"Best mIoU: {best_miou:.4f}")
    
    return model, history

def plot_training_progress(history, epoch):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # mIoU plot
    axes[0, 1].plot(history['train_miou'], label='Train mIoU')
    axes[0, 1].plot(history['val_miou'], label='Val mIoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('Training and Validation mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy plot
    axes[1, 0].plot(history['train_acc'], label='Train Accuracy')
    axes[1, 0].plot(history['val_acc'], label='Val Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training and Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate plot
    axes[1, 1].plot(history['learning_rates'], label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(config.PLOT_DIR, f'training_progress_epoch_{epoch}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Training plot saved to {plot_path}")

# Inference Function
def predict_image(model, image_path, device='cpu'):
    """Predict segmentation for a single image"""
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as validation
    transform = get_val_transform()
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Create colored segmentation
    colormap = create_pascal_label_colormap()
    colored_mask = colormap[prediction]
    
    # Blend with original
    original_resized = cv2.resize(image, (512, 512))
    blended = cv2.addWeighted(original_resized, 0.6, colored_mask, 0.4, 0)
    
    return {
        'original': original_resized,
        'mask': prediction,
        'colored_mask': colored_mask,
        'blended': blended
    }

def create_pascal_label_colormap():
    """Creates PASCAL VOC colormap"""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    col