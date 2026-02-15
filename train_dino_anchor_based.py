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
from dinov3.hub import backbones as dinov3_backbones
from safetensors.torch import load_file



# ==========================
# CONFIG
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1
LR = 1e-4
IMG_SIZE=500
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 8
SAVE_DIR = "checkpoints_anchor_based"
Path(SAVE_DIR).mkdir(exist_ok=True)


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
# def load_dino(weights_path):
#     import dinov3

#     model = dinov3_backbones.dinov3_vitl16(pretrain=False)
#     pretrained_weights = load_file(str(weights_path), device="cpu")
#     # state = torch.load(weights_path, map_location="cpu")
#     model.load_state_dict(pretrained_weights,strict=False)

#     model.eval()

#     for p in model.parameters():
#         p.requires_grad = False

#     return model

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

    model.load_state_dict(state_dict, strict=False)  # important

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model



# ==========================
# MINI FPN
# ==========================
class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        p3 = self.lateral(x)
        p4 = self.down(p3)
        p5 = self.down(p4)
        return [p3, p4, p5]

# ==========================
# RETINANET STYLE HEAD
# ==========================
class AnchorHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )

        self.box_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4, 1)
        )

    def forward(self, features):
        cls_outputs = []
        box_outputs = []

        for f in features:
            cls_outputs.append(self.cls_head(f))
            box_outputs.append(self.box_head(f))

        return cls_outputs, box_outputs

# ==========================
# FULL MODEL
# ==========================
class DinoAnchorDetector(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.backbone = dino
        self.fpn = SimpleFPN(1024, 256)
        self.head = AnchorHead(256, NUM_CLASSES)

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)['x_norm_patchtokens']

        B, N, C = tokens.shape
        H = W = int(np.sqrt(N))
        feat = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        features = self.fpn(feat)
        cls_out, box_out = self.head(features)

        return cls_out, box_out

# ==========================
# LOSS
# ==========================
def compute_loss(cls_out, box_out, targets):
    loss_cls = 0
    loss_box = 0

    for cls_f, box_f in zip(cls_out, box_out):
        loss_cls += cls_f.mean()
        loss_box += box_f.mean()

    return loss_cls + loss_box

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
# TRAIN LOOP
# ==========================
def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for imgs, targets in tqdm(train_loader):
            imgs = imgs.to(DEVICE)

            cls_out, box_out = model(imgs)
            loss = compute_loss(cls_out, box_out, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                cls_out, box_out = model(imgs)
                loss = compute_loss(cls_out, box_out, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            patience_counter = 0
            print("✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    dino = load_dino("dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth").to(DEVICE)
    model = DinoAnchorDetector(dino).to(DEVICE)

    train_ds = YoloDetectDataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN, augment=True)
    val_ds = YoloDetectDataset(IMG_DIR_VAL, LBL_DIR_VAL, augment=False)
    test_ds = YoloDetectDataset(IMG_DIR_TEST, LBL_DIR_TEST, augment=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    if len(train_ds) == 0:
        raise ValueError("No training images found!")

    # Dummy loaders (replace with real dataset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    train(model, train_loader, val_loader)
