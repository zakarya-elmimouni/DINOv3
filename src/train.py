"""
Training script for DINOv3 object detection
"""

import os
import time
import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
from .model import create_model
from .loss import DetectionLoss
from .dataset import create_dataloaders
from torch.amp import autocast, GradScaler
from .metrics import DetectionMetrics, plot_training_curves
from .utils import AverageMeter, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        self.train_loader, self.val_loader, self.num_classes = create_dataloaders(
            **config['dataset'], batch_size=config['optimizer']['batch_size'],
            img_size=config['model']['img_size'], num_workers=config['num_workers']
        )
        
        self.model = create_model(
            num_classes=self.num_classes,
            backbone=config['model']['backbone'],
            backbone_weights_path=config['model']['backbone_weights_path'],
            adapter_dim=config['model']['adapter_dim'],
            num_queries=config['model']['num_queries'],
            freeze_backbone=config['model']['freeze_backbone']
        ).to(self.device)
        
        self.criterion = DetectionLoss(num_classes=self.num_classes, loss_weights=config['loss_weights'])
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params, lr=config['optimizer']['learning_rate'], 
            weight_decay=config['optimizer']['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config['optimizer']['epochs'], 
            eta_min=config['optimizer']['learning_rate'] * 0.01
        )
        
        self.scaler = GradScaler('cuda', enabled=config['use_amp'])
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': [],
            'train_loss_components': []
        }
        
        # Early stopping
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        self.early_stop = False
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Early stopping patience: {self.patience} epochs")

    def train_epoch(self, epoch):
        self.model.train()
        if self.config['model']['freeze_backbone']: self.model.backbone.eval()
        
        losses = AverageMeter()
        loss_components = {'class': 0, 'bbox': 0, 'giou': 0, 'objectness': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['optimizer']['epochs']} [Train]")
        
        # Normalization factor to convert absolute pixel coords to [0, 1] range
        img_size = self.config['model']['img_size']
        whwh = torch.tensor([img_size, img_size, img_size, img_size], device=self.device)
        
        for images, boxes_list, labels_list in pbar:
            images = images.to(self.device)
            
            # ⭐ FIX: Normalize ground truth boxes from pixel coords [0, img_size] to [0, 1]
            gt_boxes_normalized = [b.to(self.device) / whwh for b in boxes_list]
            gt_labels = [l.to(self.device) for l in labels_list]
            
            with autocast('cuda', enabled=self.config['use_amp']):
                pred_boxes, pred_classes, pred_objectness = self.model(images)
                loss_dict = self.criterion(pred_boxes, pred_classes, pred_objectness, gt_boxes_normalized, gt_labels)
                loss = loss_dict['total']
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            losses.update(loss.item(), images.size(0))
            
            # Track loss components
            for key in loss_components.keys():
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()
            num_batches += 1
            
            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items() if torch.is_tensor(v)})
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= max(num_batches, 1)
        
        return losses.avg, loss_components

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        metrics = DetectionMetrics(self.num_classes)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['optimizer']['epochs']} [Val]")
        
        # Normalization factor to convert absolute pixel coords to [0, 1] range
        img_size = self.config['model']['img_size']
        whwh = torch.tensor([img_size, img_size, img_size, img_size], device=self.device)
        
        for images, boxes_list, labels_list in pbar:
            images = images.to(self.device)
            
            # ⭐ FIX: Normalize ground truth boxes from pixel coords [0, img_size] to [0, 1]
            gt_boxes_normalized = [b.to(self.device) / whwh for b in boxes_list]
            gt_labels = [l.to(self.device) for l in labels_list]
            
            with autocast('cuda', enabled=self.config['use_amp']):
                pred_boxes, pred_classes, pred_objectness = self.model(images)
                loss_dict = self.criterion(pred_boxes, pred_classes, pred_objectness, gt_boxes_normalized, gt_labels)
            
            losses.update(loss_dict['total'].item(), images.size(0))
            
            # Update metrics
            metrics.update(pred_boxes, pred_classes, pred_objectness, gt_boxes_normalized, gt_labels)
            
            pbar.set_postfix({'val_loss': f"{losses.avg:.4f}"})
        
        # Compute final metrics
        final_metrics = metrics.compute()
        
        # Plot confusion matrix
        metrics.plot_confusion_matrix(self.checkpoint_dir / f'confusion_matrix_epoch_{epoch+1}.png')
        
        return losses.avg, final_metrics
    
    def train(self):
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        
        for epoch in range(self.start_epoch, self.config['optimizer']['epochs']):
            if self.early_stop:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs")
                break
            
            start_time = time.time()
            
            # Train
            train_loss, loss_components = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['learning_rate'].append(current_lr)
            self.history['train_loss_components'].append(loss_components)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['optimizer']['epochs']} Summary")
            print(f"{'='*60}")
            print(f"Time: {time.time()-start_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1_score']:.4f}")
            print(f"TP: {val_metrics['true_positives']} | FP: {val_metrics['false_positives']} | FN: {val_metrics['false_negatives']}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"{'='*60}\n")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"✅ New best model! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"⚠️  No improvement. Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    self.early_stop = True
            
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'config': self.config,
                'num_classes': self.num_classes,
                'val_metrics': val_metrics
            }, is_best, self.checkpoint_dir)
            
            # Save history and results
            with open(self.checkpoint_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Save results summary
            results = {
                'best_val_loss': self.best_val_loss,
                'best_epoch': epoch + 1 - self.patience_counter if not is_best else epoch + 1,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'final_metrics': val_metrics,
                'total_epochs': epoch + 1,
                'stopped_early': self.early_stop
            }
            with open(self.checkpoint_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Plot training curves
            try:
                plot_training_curves(self.history, self.checkpoint_dir)
            except Exception as e:
                print(f"Warning: Could not plot training curves: {e}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total epochs: {epoch + 1}")
        print(f"Checkpoints and visualizations saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")