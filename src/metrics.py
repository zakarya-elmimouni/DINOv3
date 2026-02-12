"""
Metrics calculation and visualization for object detection
"""
import json
import torch
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class DetectionMetrics:
    def __init__(self, num_classes, class_names=None, iou_threshold=0.5):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics for new epoch"""
        self.all_pred_labels = []
        self.all_true_labels = []
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
    
    @torch.no_grad()
    def update(self, pred_boxes, pred_classes, pred_objectness, gt_boxes_list, gt_labels_list):
        """
        Update metrics with a batch of predictions
        
        Args:
            pred_boxes: [B, N, 4] predicted boxes (normalized [0,1])
            pred_classes: [B, N, num_classes+1] predicted class logits
            pred_objectness: [B, N, 1] objectness scores
            gt_boxes_list: list of [M, 4] ground truth boxes per image
            gt_labels_list: list of [M] ground truth labels per image
        """
        batch_size = pred_boxes.shape[0]
        
        for i in range(batch_size):
            # Get predictions for this image
            obj_scores = pred_objectness[i].sigmoid().squeeze(-1)  # [N]
            class_probs = torch.softmax(pred_classes[i], dim=-1)  # [N, num_classes+1]
            
            # Filter predictions by objectness threshold
            obj_mask = obj_scores > 0.5
            if obj_mask.sum() == 0:
                # No detections, all GT are false negatives
                self.false_negatives += len(gt_labels_list[i])
                continue
            
            pred_boxes_i = pred_boxes[i][obj_mask]  # [K, 4]
            pred_class_probs = class_probs[obj_mask]  # [K, num_classes+1]
            
            # Get predicted class (excluding background class)
            pred_labels_i = pred_class_probs[:, :-1].argmax(dim=-1)  # [K]
            
            # Get ground truth for this image
            gt_boxes_i = gt_boxes_list[i]  # [M, 4]
            gt_labels_i = gt_labels_list[i]  # [M]
            
            if len(gt_boxes_i) == 0:
                # No ground truth, all predictions are false positives
                self.false_positives += len(pred_labels_i)
                self.all_pred_labels.extend(pred_labels_i.cpu().numpy())
                continue
            
            # Compute IoU between all pred and gt boxes
            ious = self._box_iou(pred_boxes_i, gt_boxes_i)  # [K, M]
            
            # Match predictions to ground truth
            matched_gt = set()
            for pred_idx in range(len(pred_boxes_i)):
                max_iou, gt_idx = ious[pred_idx].max(dim=0)
                gt_idx = gt_idx.item()
                
                pred_label = pred_labels_i[pred_idx].item()
                
                if max_iou >= self.iou_threshold and gt_idx not in matched_gt:
                    # Matched detection
                    gt_label = gt_labels_i[gt_idx].item()
                    matched_gt.add(gt_idx)
                    
                    self.all_pred_labels.append(pred_label)
                    self.all_true_labels.append(gt_label)
                    
                    if pred_label == gt_label:
                        self.true_positives += 1
                    else:
                        self.false_positives += 1
                else:
                    # False positive (no match or wrong IoU)
                    self.false_positives += 1
                    self.all_pred_labels.append(pred_label)
            
            # Count unmatched ground truth as false negatives
            self.false_negatives += len(gt_labels_i) - len(matched_gt)
            for gt_idx in range(len(gt_labels_i)):
                if gt_idx not in matched_gt:
                    self.all_true_labels.append(gt_labels_i[gt_idx].item())
    
    def _box_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes
        boxes: [x1, y1, x2, y2] format, normalized [0, 1]
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [K, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [K, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [K, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [K, M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        return iou
    
    def compute(self):
        """Compute final metrics"""
        metrics = {}
        
        # Precision, Recall, F1
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['true_positives'] = self.true_positives
        metrics['false_positives'] = self.false_positives
        metrics['false_negatives'] = self.false_negatives
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        if len(self.all_true_labels) == 0 or len(self.all_pred_labels) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        # Pad to same length if needed
        max_len = max(len(self.all_true_labels), len(self.all_pred_labels))
        true_labels = self.all_true_labels + [-1] * (max_len - len(self.all_true_labels))
        pred_labels = self.all_pred_labels + [-1] * (max_len - len(self.all_pred_labels))
        
        # Filter out -1 (unmatched)
        valid_mask = [(t >= 0 and p >= 0) for t, p in zip(true_labels, pred_labels)]
        true_labels = [t for t, v in zip(true_labels, valid_mask) if v]
        pred_labels = [p for p, v in zip(pred_labels, valid_mask) if v]
        
        if len(true_labels) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        cm = confusion_matrix(true_labels, pred_labels, 
                            labels=list(range(self.num_classes)))
        return cm
    
    def plot_confusion_matrix(self, save_path):
        """Plot and save confusion matrix"""
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='2f', cmap='Blues', # fmt='d' for integer counts, '2f' for normalized
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def plot_training_curves(history, save_dir):
    """Plot training curves"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Precision/Recall
    if 'val_precision' in history and len(history['val_precision']) > 0:
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
        plt.plot(epochs, history['val_recall'], 'orange', label='Recall', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
    
    # Plot F1 Score
    if 'val_f1' in history and len(history['val_f1']) > 0:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['val_f1'], 'm-', label='F1 Score', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot detailed loss components if available
    if 'train_loss_components' in history and len(history['train_loss_components']) > 0:
        plt.figure(figsize=(12, 8))
        
        components = ['class', 'bbox', 'giou', 'objectness']
        for idx, comp in enumerate(components, 1):
            plt.subplot(2, 2, idx)
            comp_losses = [epoch_loss.get(comp, 0) for epoch_loss in history['train_loss_components']]
            if comp_losses:
                plt.plot(epochs, comp_losses, linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{comp.capitalize()} Loss')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'loss_components.png', dpi=150, bbox_inches='tight')
        plt.close()
