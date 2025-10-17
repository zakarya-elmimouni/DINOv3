"""
Dataset module for YOLO format object detection data
"""
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

weights_file = Path('dataset/sample_weights.json')

class YOLODetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=518, augment=False, num_classes=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = sorted([p for ext in image_extensions for p in self.images_dir.glob(ext)])
        
        self.valid_samples = [
            (img_path, self.labels_dir / f"{img_path.stem}.txt")
            for img_path in self.image_files
            if (self.labels_dir / f"{img_path.stem}.txt").exists()
        ]
        
        if not self.valid_samples:
            raise ValueError(f"No valid image-label pairs found in {images_dir} and {labels_dir}")
        print(f"Found {len(self.valid_samples)} valid image-label pairs.")
        
        self.num_classes = num_classes if num_classes is not None else self._detect_num_classes()
        print(f"Number of classes detected/set: {self.num_classes}")
        
        self.transform = self._get_transforms()
    
    def _detect_num_classes(self):
        max_class_id = -1
        print("Auto-detecting number of classes...")
        for _, label_path in tqdm(self.valid_samples, desc="Scanning labels"):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip(): max_class_id = max(max_class_id, int(line.split()[0]))
            except Exception as e: print(f"Warning: Error reading {label_path}: {e}")
        if max_class_id == -1: raise ValueError("Could not detect any classes.")
        return max_class_id + 1

    def _get_transforms(self):
        bbox_params = A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'], 
            min_visibility=0.1,
            min_area=0.0,
            clip=True
        )
        if self.augment:
            return A.Compose([
                A.OneOf([
                    A.RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                    A.Resize(height=self.img_size, width=self.img_size),
                ], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=bbox_params)
        else:
            return A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=bbox_params)

    def _load_yolo_annotations(self, label_path):
        boxes, labels = [], []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    class_id = int(parts[0])
                    x_c, y_c, w, h = [float(c) for c in parts[1:5]]
                    
                    x_c = np.clip(x_c, 0, 1)
                    y_c = np.clip(y_c, 0, 1)
                    w = np.clip(w, 0, 1)
                    h = np.clip(h, 0, 1)
                    
                    if w > 0 and h > 0:
                        labels.append(class_id)
                        boxes.append([x_c, y_c, w, h])

        except Exception as e: 
            print(f"Error reading {label_path}: {e}")
            
        return boxes, labels
    
    def __len__(self): return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.valid_samples[idx]
        image = cv2.imread(str(img_path))
        if image is None: return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = self._load_yolo_annotations(label_path)
        
        # Skip if no valid boxes
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))
        
        try:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image, boxes, labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            
            # Clip boxes to valid range [0, 1] after transformation to handle floating-point errors
            boxes_clipped = []
            labels_clipped = []
            for box, label in zip(boxes, labels):
                x_c, y_c, w, h = box
                x_c = np.clip(x_c, 0.0, 1.0)
                y_c = np.clip(y_c, 0.0, 1.0)
                w = np.clip(w, 0.0, 1.0)
                h = np.clip(h, 0.0, 1.0)
                
                # Only keep valid boxes
                if w > 0 and h > 0:
                    boxes_clipped.append([x_c, y_c, w, h])
                    labels_clipped.append(label)
            
            boxes = boxes_clipped
            labels = labels_clipped
            
            # Skip if all boxes were filtered out
            if len(boxes) == 0:
                return self.__getitem__((idx + 1) % len(self))
                
        except Exception as e:
            print(f"Warning: Transform failed on {img_path.name}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        boxes_abs = [[(c[0]-c[2]/2)*self.img_size, (c[1]-c[3]/2)*self.img_size, (c[0]+c[2]/2)*self.img_size, (c[1]+c[3]/2)*self.img_size] for c in boxes]
        return image, torch.tensor(boxes_abs, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    return torch.stack(images, 0), list(boxes), list(labels)

def create_dataloaders(train_img_dir, train_label_dir, val_img_dir, val_label_dir, batch_size=8, img_size=518, num_workers=4, num_classes=None):
    train_ds = YOLODetectionDataset(train_img_dir, train_label_dir, img_size, True, num_classes)
    val_ds = YOLODetectionDataset(val_img_dir, val_label_dir, img_size, False, train_ds.num_classes)
    
    if weights_file.exists():
        with open(weights_file, 'r') as f:
            weights_data = json.load(f)
        sample_weights = weights_data['sample_weights']

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds, batch_size, 
            sampler=sampler,
            num_workers=num_workers, 
            collate_fn=collate_fn, 
            pin_memory=True, 
            persistent_workers=num_workers > 0, 
            drop_last=True
        )
    else:
        train_loader = DataLoader(train_ds, batch_size, True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=num_workers > 0, drop_last=True)
    
    val_loader = DataLoader(val_ds, batch_size*2, False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=num_workers > 0)
    
    print(f"\n--- Dataset Stats ---\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Classes: {train_ds.num_classes}\n---------------------\n")
    return train_loader, val_loader, train_ds.num_classes