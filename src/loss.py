"""
Loss functions for object detection training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union

def generalized_box_iou(boxes1, boxes2):
    iou = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - (iou * (area1[:, None] + area2 - 1e-6))
    return iou - (area_c - union) / (area_c + 1e-6)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, pred_boxes, pred_classes, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            return []
        
        pred_probs = pred_classes.softmax(-1)
        cost_class = -pred_probs[:, gt_labels]
        
        # Scale boxes to [0,1] before L1 and GIoU cost
        h, w = 1.0, 1.0 # Assuming normalized coordinates from here
        
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)
        cost_giou = -generalized_box_iou(pred_boxes, gt_boxes)
        
        C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou).cpu()
        
        pred_idx, gt_idx = linear_sum_assignment(C)
        return list(zip(pred_idx, gt_idx))

class DetectionLoss(nn.Module):
    def __init__(self, num_classes, matcher=None, loss_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.loss_weights = loss_weights or {'class': 2.0, 'bbox': 5.0, 'giou': 2.0, 'obj': 1.0}
        
    def forward(self, pred_boxes_xyxy, pred_classes, pred_objectness, gt_boxes_list, gt_labels_list):
        batch_size, device = pred_boxes_xyxy.shape[0], pred_boxes_xyxy.device
        
        total_loss = 0
        loss_dict = {'total': 0, 'class': 0, 'bbox': 0, 'giou': 0, 'objectness': 0}
        num_total_gt = 0

        for i in range(batch_size):
            pred_b, pred_c, pred_o = pred_boxes_xyxy[i], pred_classes[i], pred_objectness[i].squeeze(-1)
            gt_b, gt_l = gt_boxes_list[i].to(device), gt_labels_list[i].to(device)
            num_gt = gt_b.shape[0]
            num_total_gt += num_gt
            
            # Match predictions to ground truth
            matches = self.matcher(pred_b, pred_c, gt_b, gt_l)
            
            # --- Objectness Loss ---
            obj_targets = torch.zeros_like(pred_o)
            if matches:
                pred_idx = torch.tensor([m[0] for m in matches], device=device)
                obj_targets[pred_idx] = 1.0
            total_loss += self.loss_weights['obj'] * F.binary_cross_entropy_with_logits(pred_o, obj_targets)
            loss_dict['objectness'] += F.binary_cross_entropy_with_logits(pred_o, obj_targets, reduction='sum')

            # --- Classification and Box Losses (only for matched pairs) ---
            if not matches:
                continue

            pred_idx, gt_idx = zip(*matches)
            pred_idx = torch.tensor(pred_idx, device=device)
            gt_idx = torch.tensor(gt_idx, device=device)

            # Class loss
            cls_targets = torch.full((pred_c.shape[0],), self.num_classes, dtype=torch.long, device=device)
            cls_targets[pred_idx] = gt_l[gt_idx]
            total_loss += self.loss_weights['class'] * F.cross_entropy(pred_c, cls_targets, reduction='mean')
            loss_dict['class'] += F.cross_entropy(pred_c, cls_targets, reduction='sum')
            
            # Box losses
            matched_pred_boxes = pred_b[pred_idx]
            matched_gt_boxes = gt_b[gt_idx]
            
            bbox_loss = F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='sum')
            total_loss += self.loss_weights['bbox'] * bbox_loss / num_gt
            loss_dict['bbox'] += bbox_loss

            giou_loss = (1 - torch.diag(generalized_box_iou(matched_pred_boxes, matched_gt_boxes))).sum()
            total_loss += self.loss_weights['giou'] * giou_loss / num_gt
            loss_dict['giou'] += giou_loss

        # Normalize losses
        if num_total_gt > 0:
            for k in ['class', 'bbox', 'giou', 'objectness']:
                loss_dict[k] /= num_total_gt
        loss_dict['total'] = total_loss / batch_size
        return loss_dict