"""
Model training script
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from config import Config
from utils import PascalVOCDataset, SimpleSegmentationModel, SegmentationMetrics

class SegmentationTrainer:
    """Trainer for semantic segmentation model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics = None
        
        # Training state
        self.current_epoch = 0
        self.best_iou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Create directories
        Config.ensure_directories()
        
        # Setup paths
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def _get_device(self):
        """Get available device"""
        if self.config.USE_GPU and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU")
        return device
    
    def setup_model(self):
        """Initialize model, optimizer, and loss function"""
        # Model
        self.model = SimpleSegmentationModel(
            num_classes=self.config.NUM_CLASSES
        ).to(self.device)
        
        # Loss function (CrossEntropy + Dice Loss)
        self.criterion = self._create_criterion()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Metrics
        self.metrics = SegmentationMetrics(self.config.NUM_CLASSES)
        
        print(f"✅ Model initialized with {self.count_parameters():,} parameters")
    
    def _create_criterion(self):
        """Create loss function"""
        class_weight = None
        
        # Optional: Add class weights for imbalanced datasets
        if hasattr(self.config, 'CLASS_WEIGHTS'):
            class_weight = torch.tensor(
                self.config.CLASS_WEIGHTS, device=self.device
            )
        
        # CrossEntropy Loss
        ce_loss = nn.CrossEntropyLoss(
            weight=class_weight,
            ignore_index=255  # Ignore void class if present
        )
        
        return ce_loss
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def load_datasets(self, dataset_path: str):
        """Load training and validation datasets"""
        print(f"📂 Loading datasets from: {dataset_path}")
        
        # Dataset transformations
        train_transform = transforms.Compose([
            # Add data augmentation for training
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        ])
        
        # Training dataset
        train_dataset = PascalVOCDataset(
            image_dir=os.path.join(dataset_path, "JPEGImages"),
            mask_dir=os.path.join(dataset_path, "SegmentationClass"),
            split="train",
            image_size=self.config.IMAGE_SIZE
        )
        
        # Validation dataset
        val_dataset = PascalVOCDataset(
            image_dir=os.path.join(dataset_path, "JPEGImages"),
            mask_dir=os.path.join(dataset_path, "SegmentationClass"),
            split="val",
            image_size=self.config.IMAGE_SIZE
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"📊 Training: {len(train_dataset)} images")
        print(f"📊 Validation: {len(val_dataset)} images")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        self.metrics.reset()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate predictions
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                self.metrics.update(preds, masks_np)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        epoch_metrics = self.metrics.get_scores()
        
        return avg_loss, epoch_metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc="Validation"):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Calculate predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                self.metrics.update(preds, masks_np)
        
        # Calculate metrics
        avg_loss = epoch_loss / len(val_loader)
        val_metrics = self.metrics.get_scores()
        
        return avg_loss, val_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'config': self.config.__dict__
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved best model (IoU: {self.best_iou:.4f})")
        
        # Latest model
        latest_path = Path(self.config.MODEL_FOLDER) / "latest_model.pth"
        torch.save(self.model.state_dict(), latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_iou = checkpoint['best_iou']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"   Best IoU: {self.best_iou:.4f}")
        
        return True
    
    def train(self, dataset_path: str, epochs: int = None):
        """Main training loop"""
        # Setup
        self.setup_model()
        
        # Load datasets
        train_loader, val_loader = self.load_datasets(dataset_path)
        
        # Use provided epochs or config epochs
        if epochs is None:
            epochs = self.config.NUM_EPOCHS
        
        print(f"🚀 Starting training for {epochs} epochs")
        print(f"📈 Target metrics will be saved to: {self.log_dir}")
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            start_time = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            train_time = time.time() - start_time
            
            # Validate
            start_time = time.time()
            val_loss, val_metrics = self.validate(val_loader)
            val_time = time.time() - start_time
            
            # Update learning rate
            self.scheduler.step(val_metrics["mean_iou"])
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Time: Train={train_time:.1f}s, Val={val_time:.1f}s")
            print(f"   Loss: Train={train_loss:.4f}, Val={val_loss:.4f}")
            print(f"   Train Pixel Acc: {train_metrics['pixel_accuracy']:.4f}")
            print(f"   Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
            print(f"   Val mIoU: {val_metrics['mean_iou']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics["mean_iou"] > self.best_iou
            if is_best:
                self.best_iou = val_metrics["mean_iou"]
            
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Save training log
            self.save_training_log(epoch + 1)
        
        print(f"\n🎉 Training completed!")
        print(f"🏆 Best validation IoU: {self.best_iou:.4f}")
        
        # Save final model
        final_path = Path(self.config.MODEL_FOLDER) / "final_model.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"💾 Final model saved to: {final_path}")
        
        return self.model
    
    def save_training_log(self, epoch: int):
        """Save training log to JSON"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "val_loss": self.val_losses[-1] if self.val_losses else None,
            "train_metrics": self.train_metrics[-1] if self.train_metrics else None,
            "val_metrics": self.val_metrics[-1] if self.val_metrics else None,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "best_iou": self.best_iou
        }
        
        log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_training_status(self):
        """Get current training status for web interface"""
        if not self.train_losses:
            return {
                "status": "not_started",
                "epoch": 0,
                "total_epochs": self.config.NUM_EPOCHS,
                "progress": 0,
                "current_loss": 0,
                "current_iou": 0
            }
        
        progress = (len(self.train_losses) / self.config.NUM_EPOCHS) * 100
        
        return {
            "status": "training",
            "epoch": len(self.train_losses),
            "total_epochs": self.config.NUM_EPOCHS,
            "progress": progress,
            "current_loss": self.train_losses[-1] if self.train_losses else 0,
            "current_iou": self.val_metrics[-1]["mean_iou"] if self.val_metrics else 0,
            "best_iou": self.best_iou
        }

def train_model_background(config_params: dict, callback=None):
    """Background training function for web interface"""
    try:
        # Update config with provided parameters
        for key, value in config_params.items():
            if hasattr(Config, key):
                setattr(Config, key, value)
        
        # Initialize trainer
        trainer = SegmentationTrainer(Config)
        
        # Find dataset
        from setup_data import DatasetSetup
        dataset_path = DatasetSetup.find_dataset()
        
        if not dataset_path:
            dataset_path = DatasetSetup.setup()
        
        if not dataset_path:
            if callback:
                callback("error", "Dataset not found. Please download it first.")
            return None
        
        # Train
        model = trainer.train(str(dataset_path), config_params.get('epochs', Config.NUM_EPOCHS))
        
        if callback:
            callback("complete", {
                "message": "Training completed successfully!",
                "best_iou": trainer.best_iou,
                "final_loss": trainer.val_losses[-1] if trainer.val_losses else 0
            })
        
        return model
        
    except Exception as e:
        if callback:
            callback("error", str(e))
        raise