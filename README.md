# DINOv3 Object Detection

A production-ready object detection implementation using Facebook Research's DINOv3 as a backbone. This repository extends the base DINOv3 with a complete training pipeline, comprehensive metrics tracking, and automatic model evaluation.

## 🎯 Overview

This project adapts the powerful DINOv3 vision transformer for object detection tasks using a DETR-style architecture. It includes:

- **Flexible Backbone Support**: Use any DINOv3 variant (small/base/large)
- **End-to-End Training Pipeline**: From data loading to model evaluation
- **Advanced Metrics**: Precision, Recall, F1, Confusion Matrices
- **Early Stopping**: Automatic training termination with patience
- **Comprehensive Visualizations**: Training curves, loss components, per-class performance
- **Production Ready**: Best practices for checkpointing, logging, and monitoring

## 📁 Project Structure

```text
DINOv3/
├── src/
│   ├── dataset.py          # YOLO format dataset loading with augmentations
│   ├── model.py            # DINOv3 detector architecture
│   ├── loss.py             # Detection loss (classification + bbox + GIoU)
│   ├── train.py            # Training loop with metrics and early stopping
│   ├── inference.py        # Model inference utilities
│   ├── metrics.py          # Precision/Recall/F1 and visualization
│   └── utils.py            # Helper functions and checkpointing
│
├── dinov3/                 # Original DINOv3 library
├── dataset/                # Your training data
│   ├── train/
│   │   ├── images/
│   │   └── labels/         # YOLO format (.txt)
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       └── images/
│
├── weights/                # Pretrained DINOv3 backbones
├── checkpoints/            # Training outputs (auto-created)
├── config.yaml             # Training configuration
└── run_train.py            # Main training script
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3

# Install dependencies
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv add albumentations opencv-python scipy pyyaml safetensors
uv add scikit-learn matplotlib seaborn requests
```

### 2. Prepare Your Dataset

Organize your data in YOLO format:

```text
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

**YOLO Label Format** (`label.txt`):

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1].

### 3. Download Pretrained Weights

```bash
# Option 1: Use the download script
python download_weights.py

# Option 2: Manual download
# Download from: https://dl.fbaipublicfiles.com/dinov3/
# Place in: weights/dinov3-vitl16-pretrain-lvd1689m.pth
```

### 4. Configure Training

Edit `config.yaml`:

```yaml
model:
  backbone: 'dinov3_vitl16'  # Options: dinov3_vits16, dinov3_vitb16, dinov3_vitl16
  backbone_weights_path: 'weights/dinov3-vitl16-pretrain-lvd1689m.pth'
  freeze_backbone: False     # False for fine-tuning, True for feature extraction
  adapter_dim: 256
  num_queries: 100
  img_size: 518

optimizer:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001       # Higher for training from scratch
  weight_decay: 0.0001

loss_weights:
  class: 2.0
  bbox: 5.0
  giou: 2.0
  obj: 1.0

patience: 10  # Early stopping patience
```

### 5. Train

```bash
uv run python run_train.py
```

## 📊 Training Output

During training, you'll see detailed metrics:

```text
============================================================
Epoch 5/50 Summary
============================================================
Time: 428.6s
Train Loss: 2.8154 | Val Loss: 2.7231
Precision: 0.7523 | Recall: 0.6841 | F1: 0.7163
TP: 248 | FP: 82 | FN: 114
Learning Rate: 0.000950
============================================================

✅ New best model! Val Loss: 2.7231
```

### Automatic Outputs

All outputs are saved to `checkpoints/`:

```text
checkpoints/
├── best_model.pth                    # Best model (lowest val loss)
├── last_checkpoint.pth               # Most recent checkpoint
├── results.json                      # Final training summary
├── history.json                      # Complete training history
├── training_curves.png               # Loss, Precision, Recall, F1 plots
├── loss_components.png               # Detailed loss breakdown
└── confusion_matrix_epoch_*.png      # Per-epoch confusion matrices
```

## 🎨 Features

### 1. **Robust Data Loading**

- Automatic YOLO format parsing
- Advanced augmentations (Albumentations v2.0+)
- Bounding box validation and clipping
- Coordinate space normalization
- Automatic class detection

### 2. **Flexible Architecture**

- Support for all DINOv3 variants (ViT-S/B/L)
- Adapter layers for domain adaptation
- DETR-style detection head
- Configurable number of queries

### 3. **Comprehensive Loss Function**

- Classification loss (Cross-Entropy)
- Bounding box L1 loss
- Generalized IoU (GIoU) loss
- Objectness scoring
- Hungarian matching for optimal assignment

### 4. **Advanced Training**

- **Early Stopping**: Prevents overfitting (10 epochs patience)
- **Mixed Precision**: Faster training with AMP
- **Gradient Clipping**: Stable training
- **Cosine LR Scheduling**: Smooth learning rate decay
- **Automatic Checkpointing**: Best and latest models saved

### 5. **Rich Metrics & Visualization**

- Precision, Recall, F1 Score
- Per-class confusion matrices
- Training curve plots
- Loss component breakdown
- True/False Positive/Negative tracking

## 📈 Model Architecture

```text
Input Image (518×518×3)
        ↓
DINOv3 Backbone (ViT-L/16)
        ↓
Feature Tokens [B, N_patches, 1024]
        ↓
Spatial Adapter (MLP)
        ↓
Adapted Features [B, N_patches, 256]
        ↓
Detection Head (Cross-Attention + MLP)
        ↓
Outputs:
├── Bounding Boxes [B, 100, 4]
├── Class Logits [B, 100, num_classes+1]
└── Objectness Scores [B, 100, 1]
```

## 🔧 Key Implementation Details

### Coordinate Space Handling

The implementation carefully handles coordinate spaces:

- **Dataset**: Outputs absolute pixel coordinates [0, img_size]
- **Model**: Predicts normalized coordinates [0, 1]
- **Training**: Ground truth is normalized before loss computation

### Bounding Box Format

- **YOLO (input)**: `[x_center, y_center, width, height]` (normalized)
- **Model (output)**: `[x1, y1, x2, y2]` (normalized)
- **Loss Computation**: Both formats converted to `[x1, y1, x2, y2]` for consistency

### Data Augmentation

Powered by Albumentations v2.0+ with:

- Random resized crop
- Horizontal flip
- Color jitter
- Automatic bounding box transformation
- Robust error handling and validation

## 📋 Configuration Options

### Model Backbones

| Backbone | Parameters | Recommended Use |
|----------|-----------|-----------------|
| `dinov3_vits16` | ~22M | Fast training, limited data |
| `dinov3_vitb16` | ~86M | Balanced performance |
| `dinov3_vitl16` | ~304M | Best accuracy, requires more compute |

### Training Modes

1. **Fine-tuning (Recommended)**

   ```yaml
   backbone_weights_path: 'path/to/weights.pth'
   freeze_backbone: False
   learning_rate: 0.001
   ```

2. **Feature Extraction**

   ```yaml
   backbone_weights_path: 'path/to/weights.pth'
   freeze_backbone: True
   learning_rate: 0.0001
   ```

3. **From Scratch**

   ```yaml
   backbone_weights_path: ''
   freeze_backbone: False
   learning_rate: 0.001
   batch_size: 16  # Use smaller model
   ```

## 🐛 Troubleshooting

### Common Issues

- MemoryError during multiprocessing

```yaml
# Solution: Set num_workers to 0
num_workers: 0
```

- High bbox loss (>100)

```text
# This was a coordinate space mismatch issue - now fixed!
# The training loop automatically normalizes ground truth boxes
```

- Albumentations validation errors

```text
# Fixed with proper BboxParams and coordinate clipping
# All bounding boxes are validated and clipped to [0, 1]
```

- Weights not loading (many missing keys)

```bash
# Use the correct .pth format (not .safetensors)
python download_weights.py
```

## 📊 Performance Tips

### For Faster Training

- Use smaller backbone (`dinov3_vits16`)
- Increase batch size
- Set `num_workers > 0` (if no memory issues)
- Use mixed precision (`use_amp: True`)

### For Better Accuracy

- Use larger backbone (`dinov3_vitl16`)
- Train longer (more epochs)
- Increase image size
- Adjust loss weights for your dataset

### For Limited VRAM

- Use smaller backbone
- Reduce batch size
- Reduce image size
- Freeze backbone (`freeze_backbone: True`)

## 🎓 Training Best Practices

1. **Start Small**: Begin with `dinov3_vits16` to validate your setup
2. **Monitor Metrics**: Check precision/recall, not just loss
3. **Use Pretrained Weights**: Much better than training from scratch
4. **Patience is Key**: Early stopping will save you time
5. **Visualize Results**: Check confusion matrices for class imbalances

## 📝 Results Format

### `results.json`

```json
{
  "best_val_loss": 2.7231,
  "best_epoch": 18,
  "final_train_loss": 2.8154,
  "final_val_loss": 2.7231,
  "final_metrics": {
    "precision": 0.7523,
    "recall": 0.6841,
    "f1_score": 0.7163,
    "true_positives": 248,
    "false_positives": 82,
    "false_negatives": 114
  },
  "total_epochs": 28,
  "stopped_early": true
}
```

## 🔬 Advanced Usage

### Custom Loss Weights

Adjust based on your dataset characteristics:

```yaml
loss_weights:
  class: 2.0    # Increase if classification is poor
  bbox: 5.0     # Increase if localization is inaccurate
  giou: 2.0     # Increase for better IoU
  obj: 1.0      # Increase if too many false positives
```

### Resume Training

```python
# In run_train.py, add:
checkpoint = torch.load('checkpoints/last_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## 🤝 Contributing

Contributions are welcome! Key areas for improvement:

- Additional augmentation strategies
- Multi-scale training
- Test-time augmentation
- Model ensemble methods
- Export to ONNX/TensorRT

## 📄 License

This project extends the original [DINOv3](https://github.com/facebookresearch/dinov3) repository.

- DINOv3: Apache 2.0 License
- This Extension: See LICENSE file

## 🙏 Acknowledgments

- **Facebook Research** for the excellent DINOv3 model
- **Albumentations** for robust data augmentation
- **PyTorch** for the deep learning framework

---

Happy Training! 🚀
