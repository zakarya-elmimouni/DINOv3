import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops as ops
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os 
from PIL import Image, ImageFile
import glob
import random
import math
from dinov3.hub import backbones as dinov3_backbones
from safetensors.torch import load_file

# ==========================
# CONFIG
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1  # Nombre de classes d'objets (sans le background)
LR = 1e-4
IMG_SIZE = 500
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 8
SAVE_DIR = "checkpoints_anchor_based"
Path(SAVE_DIR).mkdir(exist_ok=True)

# Définir ces variables avant de créer le modèle
ANCHOR_SIZES = [[32, 64], [128, 256], [512]]  # Tailles pour chaque niveau
ANCHOR_RATIOS = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]  # Ratios pour chaque niveau
FEATURE_STRIDES = [4, 8, 16]  # Strides pour chaque niveau

# Paramètres des ancres
ANCHOR_SIZES = [[32, 64], [128, 256], [512]]  # Tailles pour chaque niveau FPN
ANCHOR_RATIOS = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]  # Ratios pour chaque niveau
FEATURE_STRIDES = [4, 8, 16]  # Strides correspondant aux niveaux FPN

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMG_DIR_TRAIN = "dataset/train/images"
LBL_DIR_TRAIN = "dataset/train/labels"
IMG_DIR_VAL = "dataset/val/images"
LBL_DIR_VAL = "dataset/val/labels"
IMG_DIR_TEST = "dataset/test/images"
LBL_DIR_TEST = "dataset/test/labels"

# ==========================
# LOAD DINOv3 BACKBONE
# ==========================
def load_dino(weights_path):
    from dinov3.models.vision_transformer import DinoVisionTransformer

    model = DinoVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_bias=True,
        use_rms_norm=False,
        init_values=1e-5,
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model

# ==========================
# ANCHOR GENERATOR
# ==========================
class AnchorGenerator(nn.Module):
    """
    Génère des ancres pour chaque niveau FPN
    """
    def __init__(self, sizes, ratios, feature_strides):
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.feature_strides = feature_strides
        
    def forward(self, feature_maps):
        """
        Génère des ancres pour chaque niveau de feature map
        feature_maps: list de tensors [B, C, H, W]
        Returns: liste d'ancres pour chaque niveau, chaque ancre est [x1, y1, x2, y2]
        """
        all_anchors = []
        
        for level, (feat_map, sizes, ratios, stride) in enumerate(zip(
            feature_maps, self.sizes, self.ratios, self.feature_strides)):
            
            B, C, H, W = feat_map.shape
            
            # Générer les points de la grille
            shift_x = torch.arange(0, W) * stride
            shift_y = torch.arange(0, H) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            
            # Nombre d'ancres par position = len(sizes) * len(ratios)
            num_anchors_per_pos = len(sizes) * len(ratios)
            
            # Préparer les tenseurs pour toutes les ancres
            all_anchors_level = []
            
            for size in sizes:
                for ratio in ratios:
                    # Calculer largeur et hauteur de l'ancre
                    w = size * math.sqrt(ratio)
                    h = size / math.sqrt(ratio)
                    
                    # Créer les ancres pour tous les points de la grille
                    x1 = shift_x - w / 2
                    y1 = shift_y - h / 2
                    x2 = shift_x + w / 2
                    y2 = shift_y + h / 2
                    
                    # Stack et reshape
                    anchors = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
                    all_anchors_level.append(anchors)
            
            # Concaténer toutes les ancres de ce niveau
            level_anchors = torch.cat(all_anchors_level, dim=0)  # [H*W*num_anchors_per_pos, 4]
            
            # Vérification
            expected_num = H * W * num_anchors_per_pos
            assert level_anchors.shape[0] == expected_num, \
                f"Niveau {level}: {level_anchors.shape[0]} ancres mais {expected_num} attendues"
            
            all_anchors.append(level_anchors)
        
        return all_anchors


# ==========================
# MINI FPN
# ==========================
class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.down1 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        p3 = self.lateral(x)
        p4 = self.down1(p3)
        p5 = self.down2(p4)
        return [p3, p4, p5]

# ==========================
# RETINANET STYLE HEAD
# ==========================
class AnchorHead(nn.Module):
    def __init__(self, in_channels, num_classes, sizes, ratios):
        super().__init__()
        self.num_classes = num_classes
        self.sizes = sizes
        self.ratios = ratios
        
        # Calculer le nombre d'ancres par emplacement pour chaque niveau
        self.num_anchors_per_level = []
        for level_sizes, level_ratios in zip(sizes, ratios):
            self.num_anchors_per_level.append(len(level_sizes) * len(level_ratios))
        
        # Têtes pour chaque niveau (pour gérer différents nombres d'ancres)
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        
        for num_anchors in self.num_anchors_per_level:
            # Classification head
            cls_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, num_anchors * (num_classes + 1), 1)  # +1 pour background
            )
            
            # Regression head
            box_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, num_anchors * 4, 1)
            )
            
            self.cls_heads.append(cls_head)
            self.box_heads.append(box_head)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        for cls_head, box_head in zip(self.cls_heads, self.box_heads):
            for m in [cls_head, box_head]:
                for layer in m:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=0.01)
                        nn.init.constant_(layer.bias, 0)
            
            # Initialisation spéciale pour la classification
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(cls_head[-1].bias, bias_value)

    def forward(self, features):
        cls_outputs = []
        box_outputs = []
        num_anchors_per_level = []

        for level, f in enumerate(features):
            B, C, H, W = f.shape
            num_anchors = self.num_anchors_per_level[level]
            
            # Prédictions brutes
            cls_out = self.cls_heads[level](f)  # [B, (num_classes+1) * num_anchors, H, W]
            box_out = self.box_heads[level](f)  # [B, 4 * num_anchors, H, W]
            
            # Reshape pour séparer les ancres
            cls_out = cls_out.reshape(B, num_anchors, self.num_classes + 1, H, W)
            cls_out = cls_out.permute(0, 3, 4, 1, 2)  # [B, H, W, num_anchors, num_classes+1]
            
            box_out = box_out.reshape(B, num_anchors, 4, H, W)
            box_out = box_out.permute(0, 3, 4, 1, 2)  # [B, H, W, num_anchors, 4]
            
            # Nombre total d'ancres pour ce niveau
            total_anchors = H * W * num_anchors
            num_anchors_per_level.append(total_anchors)
            
            cls_outputs.append(cls_out)
            box_outputs.append(box_out)

        return cls_outputs, box_outputs, num_anchors_per_level

# ==========================
# FULL MODEL
# ==========================
class DinoAnchorDetector(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.backbone = dino
        self.fpn = SimpleFPN(1024, 256)
        
        # S'assurer que ANCHOR_SIZES et ANCHOR_RATIOS sont définis
        global ANCHOR_SIZES, ANCHOR_RATIOS, FEATURE_STRIDES
        
        self.anchor_generator = AnchorGenerator(
            ANCHOR_SIZES, ANCHOR_RATIOS, FEATURE_STRIDES
        )
        
        self.head = AnchorHead(256, NUM_CLASSES, ANCHOR_SIZES, ANCHOR_RATIOS)
        
    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)['x_norm_patchtokens']

        B, N, C = tokens.shape
        H = W = int(np.sqrt(N))
        feat = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # FPN
        features = self.fpn(feat)
        
        # Prédictions
        cls_preds, box_preds, num_anchors_per_level = self.head(features)
        
        # Générer les ancres pour chaque niveau
        anchors_per_level = self.anchor_generator(features)
        
        # Vérification détaillée
        for i, (num_anchors, anchors, feat_map) in enumerate(zip(num_anchors_per_level, anchors_per_level, features)):
            B, C, H, W = feat_map.shape
            expected_anchors = H * W * (len(ANCHOR_SIZES[i]) * len(ANCHOR_RATIOS[i]))
            
            print(f"Niveau {i}:")
            print(f"  H={H}, W={W}")
            print(f"  sizes={ANCHOR_SIZES[i]}, ratios={ANCHOR_RATIOS[i]}")
            print(f"  ancres par position = {len(ANCHOR_SIZES[i]) * len(ANCHOR_RATIOS[i])}")
            print(f"  total ancres = {H}*{W}*{len(ANCHOR_SIZES[i]) * len(ANCHOR_RATIOS[i])} = {expected_anchors}")
            print(f"  num_anchors_per_level = {num_anchors}")
            print(f"  anchors_per_level shape = {anchors.shape}")
            print()
            
            assert num_anchors == anchors.shape[0], \
                f"Niveau {i}: {num_anchors} prédictions mais {anchors.shape[0]} ancres"
        
        return cls_preds, box_preds, anchors_per_level, num_anchors_per_level

# ==========================
# LOSS FUNCTIONS
# ==========================
def box_iou(boxes1, boxes2):
    """Calcule l'IoU entre deux ensembles de boîtes"""
    # S'assurer que les deux tenseurs sont sur le même device
    device = boxes1.device
    boxes2 = boxes2.to(device)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    
    return iou


def encode_boxes(gt_boxes, anchors):
    """Encode les boîtes GT en offsets par rapport aux ancres"""
    # S'assurer que tout est sur le même device
    device = gt_boxes.device
    anchors = anchors.to(device)
    
    anchors_wh = anchors[:, 2:] - anchors[:, :2]
    anchors_ctr = (anchors[:, 2:] + anchors[:, :2]) / 2
    
    gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
    gt_ctr = (gt_boxes[:, 2:] + gt_boxes[:, :2]) / 2
    
    # Éviter la division par zéro
    anchors_wh = anchors_wh.clamp(min=1e-6)
    
    dx = (gt_ctr[:, 0] - anchors_ctr[:, 0]) / anchors_wh[:, 0]
    dy = (gt_ctr[:, 1] - anchors_ctr[:, 1]) / anchors_wh[:, 1]
    dw = torch.log(gt_wh[:, 0] / anchors_wh[:, 0] + 1e-6)
    dh = torch.log(gt_wh[:, 1] / anchors_wh[:, 1] + 1e-6)
    
    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    return deltas


# ==========================
# FONCTION decode_boxes CORRIGÉE
# ==========================
def decode_boxes(deltas, anchors):
    """Décode les offsets en boîtes absolues"""
    # S'assurer que tout est sur le même device
    device = deltas.device
    anchors = anchors.to(device)
    
    anchors_wh = anchors[:, 2:] - anchors[:, :2]
    anchors_ctr = (anchors[:, 2:] + anchors[:, :2]) / 2
    
    dx, dy, dw, dh = deltas.unbind(1)
    
    ctr_x = dx * anchors_wh[:, 0] + anchors_ctr[:, 0]
    ctr_y = dy * anchors_wh[:, 1] + anchors_ctr[:, 1]
    w = anchors_wh[:, 0] * torch.exp(dw.clamp(max=math.log(1000)))  # Limiter l'explosion
    h = anchors_wh[:, 1] * torch.exp(dh.clamp(max=math.log(1000)))
    
    x1 = ctr_x - w / 2
    y1 = ctr_y - h / 2
    x2 = ctr_x + w / 2
    y2 = ctr_y + h / 2
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes

def compute_loss(cls_preds_per_level, box_preds_per_level, anchors_per_level, targets, device):
    """
    Calcule la perte complète en traitant chaque niveau séparément
    """
    batch_size = cls_preds_per_level[0].shape[0]
    
    # Concaténer tous les niveaux
    all_cls_preds = []
    all_box_preds = []
    all_anchors = []
    
    for level, (cls_pred, box_pred, anchors) in enumerate(zip(cls_preds_per_level, box_preds_per_level, anchors_per_level)):
        # cls_pred: [B, H, W, num_anchors_per_pos, num_classes+1]
        # box_pred: [B, H, W, num_anchors_per_pos, 4]
        # anchors: [H*W*num_anchors_per_pos, 4]
        
        B, H, W, num_anchors_per_pos, num_classes = cls_pred.shape
        
        # Reshape pour avoir [B, H*W*num_anchors_per_pos, num_classes]
        cls_pred = cls_pred.reshape(B, -1, num_classes)
        box_pred = box_pred.reshape(B, -1, 4)
        
        # S'assurer que les ancres sont sur le bon device
        anchors = anchors.to(device)
        
        all_cls_preds.append(cls_pred)
        all_box_preds.append(box_pred)
        all_anchors.append(anchors)
    
    # Concaténer tous les niveaux
    cls_pred = torch.cat(all_cls_preds, dim=1)  # [B, num_anchors_total, num_classes]
    box_pred = torch.cat(all_box_preds, dim=1)  # [B, num_anchors_total, 4]
    anchors = torch.cat(all_anchors, dim=0)     # [num_anchors_total, 4]
    
    num_anchors_total = anchors.shape[0]
    
    # Initialiser les pertes comme des tenseurs
    cls_loss_total = torch.tensor(0.0, device=device)
    box_loss_total = torch.tensor(0.0, device=device)
    num_positives = 0
    
    for b in range(batch_size):
        # Récupérer les targets pour cette image
        boxes = targets[b]['boxes'].to(device) if targets[b]['boxes'].numel() > 0 else torch.zeros((0, 4), device=device)
        labels = targets[b]['labels'].to(device) if targets[b]['labels'].numel() > 0 else torch.zeros(0, dtype=torch.long, device=device)
        
        if boxes.shape[0] == 0:
            # Pas d'écoles dans l'image
            cls_target = torch.zeros(num_anchors_total, dtype=torch.long, device=device)
            cls_loss = F.cross_entropy(cls_pred[b], cls_target, reduction='mean')
            cls_loss_total = cls_loss_total + cls_loss
            continue
        
        # Calculer IoU entre toutes les ancres et toutes les boîtes GT
        ious = box_iou(anchors, boxes)  # [num_anchors_total, num_gt]
        
        # Pour chaque ancre, la meilleure GT correspondante
        max_iou, gt_idx = ious.max(dim=1)
        
        # Seuils pour assignation
        pos_thresh = 0.5
        neg_thresh = 0.4
        
        # Créer les targets
        cls_target = torch.full((num_anchors_total,), -1, dtype=torch.long, device=device)
        
        # Ancres positives (IoU > seuil haut)
        pos_mask = max_iou >= pos_thresh
        if pos_mask.any():
            cls_target[pos_mask] = labels[gt_idx[pos_mask]] + 1  # +1 car 0 est background
        
        # Ancres négatives (IoU < seuil bas)
        neg_mask = max_iou < neg_thresh
        cls_target[neg_mask] = 0  # Classe background
        
        # Classification loss
        valid_mask = cls_target != -1
        if valid_mask.any():
            cls_loss = F.cross_entropy(
                cls_pred[b][valid_mask], 
                cls_target[valid_mask], 
                reduction='mean'
            )
            cls_loss_total = cls_loss_total + cls_loss
            
            # Box regression loss pour les ancres positives
            if pos_mask.any():
                pos_anchors = anchors[pos_mask]
                pos_gt_boxes = boxes[gt_idx[pos_mask]]
                
                # Encoder les offsets
                targets_deltas = encode_boxes(pos_gt_boxes, pos_anchors)
                pred_deltas = box_pred[b][pos_mask]
                
                # Smooth L1 loss
                box_loss = F.smooth_l1_loss(pred_deltas, targets_deltas, reduction='mean')
                box_loss_total = box_loss_total + box_loss
                num_positives += pos_mask.sum().item()
    
    # Normaliser par batch
    cls_loss_total = cls_loss_total / batch_size
    
    # Gérer le cas où il n'y a pas de positifs
    if num_positives > 0:
        box_loss_total = box_loss_total / batch_size
    else:
        box_loss_total = torch.tensor(0.0, device=device)
    
    # Perte totale
    total_loss = cls_loss_total + box_loss_total
    
    return {
        'total_loss': total_loss,
        'cls_loss': cls_loss_total,
        'box_loss': box_loss_total,
        'num_positives': num_positives
    }

# ==========================
# POST-PROCESSING (NMS)
# ==========================
def postprocess_predictions(cls_preds_per_level, box_preds_per_level, anchors_per_level, 
                           score_thresh=0.05, nms_thresh=0.5, topk=100):
    """
    Post-traite les prédictions en concaténant tous les niveaux
    """
    all_cls_pred = []
    all_box_pred = []
    all_anchors = []
    
    for cls_pred, box_pred, anchors in zip(cls_preds_per_level, box_preds_per_level, anchors_per_level):
        B, H, W, num_anchors_per_loc, num_classes = cls_pred.shape
        
        # Reshape
        cls_pred = cls_pred.reshape(-1, num_classes)
        box_pred = box_pred.reshape(-1, 4)
        
        all_cls_pred.append(cls_pred)
        all_box_pred.append(box_pred)
        all_anchors.append(anchors)
    
    cls_pred = torch.cat(all_cls_pred, dim=0)
    box_pred = torch.cat(all_box_pred, dim=0)
    anchors = torch.cat(all_anchors, dim=0)
    
    # Softmax pour obtenir des probabilités
    scores = F.softmax(cls_pred, dim=-1)  # [num_anchors_total, num_classes+1]
    
    # Ignorer la classe background (indice 0)
    scores_obj, labels = scores[:, 1:].max(dim=1)
    labels = labels + 1  # Revenir aux indices originaux
    
    # Filtrer par score
    keep = scores_obj > score_thresh
    if not keep.any():
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    
    scores_obj = scores_obj[keep]
    labels = labels[keep]
    boxes = decode_boxes(box_pred[keep], anchors[keep])
    
    # Clipper les boîtes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, IMG_SIZE)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, IMG_SIZE)
    
    # Appliquer NMS
    keep_idx = ops.nms(boxes, scores_obj, nms_thresh)
    
    # Garder seulement les top-k
    if len(keep_idx) > topk:
        keep_idx = keep_idx[:topk]
    
    return boxes[keep_idx], scores_obj[keep_idx], labels[keep_idx]

# ==========================
# DATASET (inchangé)
# ==========================
def safe_image_open(img_path):
    """
    Safely open an image file, handling truncated/corrupted files
    Returns a valid PIL Image object even for corrupted files
    """
    try:
        # First, check if file exists and has content
        if not os.path.exists(img_path):
            print(f"Warning: Image file does not exist: {img_path}")
            return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        
        if os.path.getsize(img_path) == 0:
            print(f"Warning: Image file is empty: {img_path}")
            return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        
        # Try to open and verify the image
        img = Image.open(img_path)
        
        # Try to load the image data to catch truncation errors
        try:
            img.load()
        except (OSError, IOError) as e:
            print(f"Warning: Truncated image {img_path}: {e}")
            # Try to recover by converting to RGB and copying
            img = img.convert("RGB")
        
        return img.convert("RGB")
        
    except Exception as e:
        print(f"Warning: Failed to open image {img_path}: {e}")
        # Create a blank image as fallback
        return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')


def validate_dataset_files(img_dir, lbl_dir):
    """
    Validate all images in dataset and report issues
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    valid_count = 0
    corrupted_count = 0
    
    print(f"Validating dataset in {img_dir}...")
    
    for img_path in img_paths:
        try:
            # Test image opening
            img = safe_image_open(img_path)
            img.verify()  # Verify it's a valid image
            
            # Check corresponding label file
            lbl_path = os.path.join(lbl_dir, Path(img_path).stem + ".txt")
            if os.path.exists(lbl_path):
                valid_count += 1
            else:
                print(f"Warning: No label file for {img_path}")
                
        except Exception as e:
            corrupted_count += 1
            print(f"Corrupted image: {img_path} - {e}")
    
    print(f"Dataset validation: {valid_count} valid, {corrupted_count} corrupted images")
    return valid_count, corrupted_count

def load_yolo_txt(lbl_path, img_w, img_h):
    boxes, labels = [], []
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    
    with open(lbl_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = map(float, parts)
        x1 = (cx - w/2.0) * img_w
        y1 = (cy - h/2.0) * img_h
        x2 = (cx + w/2.0) * img_w
        y2 = (cy + h/2.0) * img_h
        
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 >= img_w or y1 >= img_h or x2 <= 0 or y2 <= 0:
            continue
            
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls) + 1)  # 0 is background in detection models
    
    if not boxes:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    
    boxes = np.array(boxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, img_w)
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, img_h)
    
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    
    return boxes, labels

class YoloDetectDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, augment=True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.lbl_dir = lbl_dir
        self.augment = augment
        print(f"Found {len(self.img_paths)} images in {img_dir}")
        
        # Validate dataset files
        valid_count, corrupted_count = validate_dataset_files(img_dir, lbl_dir)
        
        # Debug: check labels
        label_counts = 0
        for img_path in self.img_paths[:10]:  # Check first 10
            lbl_path = os.path.join(self.lbl_dir, Path(img_path).stem + ".txt")
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                label_counts += 1
        print(f"Debug: {label_counts}/10 images have non-empty labels")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        p = self.img_paths[idx]
        
        # Use safe image loading - FIXED
        img = safe_image_open(p)
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        
        arr = (arr - np.array(MEAN)) / np.array(STD)
        
        flipped = False
        if self.augment and random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1, :])
            flipped = True
        
        H, W = arr.shape[:2]
        lbl_path = os.path.join(self.lbl_dir, Path(p).stem + ".txt")
        boxes, labels = load_yolo_txt(lbl_path, W, H)
        
        if flipped and boxes.shape[0] > 0:
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        
        if boxes.shape[0] == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "img_path": p,
            }
        else:
            target = {
                "boxes": torch.from_numpy(boxes).float(),
                "labels": torch.from_numpy(labels).long(),
                "image_id": torch.tensor([idx]),
                "img_path": p,
            }
        
        return img_t, target

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return torch.stack(imgs, 0), list(targets)

# ==========================
# TRAIN LOOP (mis à jour)
# ==========================
def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_box_loss = 0.0
        train_positives = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs = imgs.to(DEVICE)
            
            # S'assurer que les targets sont sur le bon device
            for t in targets:
                if 'boxes' in t and t['boxes'].numel() > 0:
                    t['boxes'] = t['boxes'].to(DEVICE)
                if 'labels' in t and t['labels'].numel() > 0:
                    t['labels'] = t['labels'].to(DEVICE)
            
            # Forward
            cls_preds, box_preds, anchors_per_level, num_anchors_per_level = model(imgs)
            
            # Compute loss
            losses = compute_loss(cls_preds, box_preds, anchors_per_level, targets, DEVICE)
            
            # Backward
            optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Convertir les tenseurs en float pour l'accumulation
            train_loss += losses['total_loss'].item()
            train_cls_loss += losses['cls_loss'].item()
            train_box_loss += losses['box_loss'].item()
            train_positives += losses['num_positives']

        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_box_loss = 0.0
        val_positives = 0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                imgs = imgs.to(DEVICE)
                
                # S'assurer que les targets sont sur le bon device
                for t in targets:
                    if 'boxes' in t and t['boxes'].numel() > 0:
                        t['boxes'] = t['boxes'].to(DEVICE)
                    if 'labels' in t and t['labels'].numel() > 0:
                        t['labels'] = t['labels'].to(DEVICE)
                
                cls_preds, box_preds, anchors_per_level, num_anchors_per_level = model(imgs)
                losses = compute_loss(cls_preds, box_preds, anchors_per_level, targets, DEVICE)
                
                val_loss += losses['total_loss'].item()
                val_cls_loss += losses['cls_loss'].item()
                val_box_loss += losses['box_loss'].item()
                val_positives += losses['num_positives']

        # Normaliser les pertes
        train_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_box_loss /= len(train_loader)
        
        val_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_box_loss /= len(val_loader)
        
        # Mettre à jour le scheduler
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Total: {train_loss:.4f}, Cls: {train_cls_loss:.4f}, Box: {train_box_loss:.4f}, Pos: {train_positives}")
        print(f"  Val   - Total: {val_loss:.4f}, Cls: {val_cls_loss:.4f}, Box: {val_box_loss:.4f}, Pos: {val_positives}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f"{SAVE_DIR}/best_model.pth")
            patience_counter = 0
            print("✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

# ==========================
# INFERENCE EXAMPLE
# ==========================
def detect_single_image(model, image_path):
    """Exemple d'inférence sur une seule image"""
    model.eval()
    
    # Charger l'image
    img = safe_image_open(image_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Prétraiter
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    # Prédire
    with torch.no_grad():
        cls_pred, box_pred, anchors = model(img_tensor)
        
    # Post-traiter
    boxes, scores, labels = postprocess_predictions(
        cls_pred[0], box_pred[0], anchors,
        score_thresh=0.3, nms_thresh=0.5
    )
    
    return boxes.cpu(), scores.cpu(), labels.cpu()

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    # Charger le modèle
    dino = load_dino("dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth").to(DEVICE)
    model = DinoAnchorDetector(dino).to(DEVICE)
    
    # Compter les paramètres
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")

    # Charger les datasets
    train_ds = YoloDetectDataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN, augment=True)
    val_ds = YoloDetectDataset(IMG_DIR_VAL, LBL_DIR_VAL, augment=False)
    test_ds = YoloDetectDataset(IMG_DIR_TEST, LBL_DIR_TEST, augment=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    if len(train_ds) == 0:
        raise ValueError("No training images found!")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    # Entraînement
    train(model, train_loader, val_loader)
    
    print("Training completed!")