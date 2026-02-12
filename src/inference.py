"""
Inference script for DINOv3 object detection
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from .model import create_model

class Inferencer:
    def __init__(self, checkpoint_path, device='cuda', img_size=500):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        config = checkpoint.get('config', {})
        num_classes = checkpoint.get('num_classes')
        if num_classes is None:
            raise ValueError("num_classes not found in checkpoint. Please specify it.")
        
        self.num_classes = num_classes
        
        print(f"Loading model with {num_classes} classes...")
        self.model = create_model(
            num_classes=num_classes,
            backbone=config.get('model', {}).get('backbone', 'dinov3_vitl16'),
            adapter_dim=config.get('model', {}).get('adapter_dim', 256),
            num_queries=config.get('model', {}).get('num_queries', 100)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"✓ Model loaded successfully. Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    @torch.no_grad()
    def predict(self, image_path, conf_threshold=0.5, nms_threshold=0.5):
        image = cv2.imread(str(image_path))
        if image is None: raise FileNotFoundError(f"Image not found at {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        results = self.model.get_predictions(image_tensor, conf_threshold, nms_threshold)[0]
        
        # Scale boxes back to original image size
        boxes = results['boxes'].cpu()
        if len(boxes) > 0:
            # Note: get_predictions already scales the boxes, so this might be redundant
            # depending on the implementation. Assuming it returns relative coordinates.
            pass # The model's get_predictions should handle scaling

        return {
            'boxes': boxes.numpy(),
            'labels': results['labels'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy()
        }
    
    def predict_batch(self, image_paths, conf_threshold=0.5, nms_threshold=0.5):
        return [
            self.predict(img_path, conf_threshold, nms_threshold)
            for img_path in tqdm(image_paths, desc="Batch Prediction")
        ]
    
    def visualize(self, image_path, predictions, class_names=None, output_path=None, line_thickness=2):
        image = cv2.imread(str(image_path))
        
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            x1, y1, x2, y2 = map(int, box)
            
            np.random.seed(int(label))
            color = tuple(np.random.randint(100, 255, 3).tolist())
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
            
            label_text = f"{class_names[label] if class_names else f'cls_{label}'}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(image, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if output_path:
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(output_path), image)
            print(f"✓ Visualization saved to {output_path}")
        
        return image