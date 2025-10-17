# 🎯 Quick Reference: Class Imbalance Fix

## The Problem (One Sentence)
Your model defaults to predicting `Class_0` because it's 2× more common, causing 94% of `Class_1` to be misclassified.

## The Solution (One Sentence)
Make rare classes cost more in the loss function and sample them more frequently during training.

---

## Files to Edit

### 1. config.yaml (2 min)
```yaml
loss_weights:
  class: 10.0  # Change from 1.0
```

### 2. src/loss.py (10 min)
```python
class DetectionLoss(nn.Module):
    def __init__(self, num_classes, matcher=None, loss_weights=None, class_weights=None):
        super().__init__()
        # ... existing code ...
        
        # ADD THIS:
        if class_weights is None:
            self.class_weights = torch.ones(num_classes + 1)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
```

```python
# In forward(), change:
total_loss += self.loss_weights['class'] * F.cross_entropy(pred_c, cls_targets)

# To:
cls_weights = self.class_weights.to(device)
total_loss += self.loss_weights['class'] * F.cross_entropy(
    pred_c, cls_targets, weight=cls_weights, reduction='mean'
)
```

### 3. src/train.py (5 min)
```python
# In Trainer.__init__, before self.criterion = DetectionLoss(...):

# ADD THIS:
import json
weights_file = Path('dataset/sample_weights.json')
if weights_file.exists():
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)
    class_weights = weights_data['class_weights']
    print(f"✓ Loaded class weights: {class_weights[:self.num_classes]}")
else:
    class_weights = None
    print("⚠️  No class weights found, using equal weights")

# MODIFY THIS:
self.criterion = DetectionLoss(
    num_classes=self.num_classes, 
    loss_weights=config['loss_weights'],
    class_weights=class_weights  # ADD THIS LINE
)
```

### 4. src/dataset.py (15 min - OPTIONAL)
```python
# At top, add import:
from torch.utils.data import WeightedRandomSampler

# In create_dataloaders(), replace train_loader creation:
weights_file = Path('dataset/sample_weights.json')
if weights_file.exists():
    import json
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
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=num_workers, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        persistent_workers=num_workers > 0, 
        drop_last=True
    )
else:
    # Keep existing DataLoader with shuffle=True
    train_loader = DataLoader(train_ds, batch_size, True, ...)
```

---

## Verification Checklist

Before training:
- [ ] `dataset/sample_weights.json` exists (run `quick_improvements.py`)
- [ ] `config.yaml` has `class: 10.0`
- [ ] `loss.py` has `class_weights` parameter
- [ ] `train.py` loads and passes class weights
- [ ] Optional: `dataset.py` uses WeightedRandomSampler

During training (check epoch 1):
- [ ] Console shows "✓ Loaded class weights"
- [ ] Confusion matrix shows Class_1 improving from 6% → 15%+

After 10 epochs:
- [ ] Class_1 accuracy > 25%
- [ ] Class_2 accuracy > 60%
- [ ] Overall F1 > 0.72

After 50 epochs:
- [ ] Class_1 accuracy > 35%
- [ ] Class_2 accuracy > 65%
- [ ] Overall F1 > 0.75

---

## What Changed in Your Model

**Before**: 
```
Loss = 10.0 × (equal_penalty_for_all_classes)
```

**After**:
```
Loss = 10.0 × (0.67×Class_0_errors + 1.37×Class_1_errors + 1.29×Class_2_errors)
```

Result: Model pays more attention to rare classes!

---

## Expected Timeline

```
Day 0 (Now):
  - Run quick_improvements.py ✅
  - Edit 3-4 files (30 min)
  
Day 1:
  - Start training overnight
  - Check epoch 1 confusion matrix next morning
  
Day 2:
  - Training complete (50 epochs)
  - Compare checkpoints-v1/ vs checkpoints/
  - If F1 > 0.75: SUCCESS! 🎉
  - If F1 < 0.73: Check if weights loaded correctly
```

---

## One-Line Commands

```powershell
# 1. Compute weights (already done)
uv run python quick_improvements.py

# 2. After editing files, start training
uv run python run_train.py

# 3. While training, watch confusion matrices
Get-ChildItem checkpoints\confusion_matrix_*.png | Sort-Object LastWriteTime | Select-Object -Last 1
```

---

## Success Metrics

| Metric | Before | Target | Stretch Goal |
|--------|--------|--------|--------------|
| Overall F1 | 0.693 | 0.750 | 0.800 |
| Class_0 Acc | 100% | 90%+ | 95%+ |
| Class_1 Acc | 6% | 35%+ | 50%+ |
| Class_2 Acc | 51% | 65%+ | 75%+ |

---

## If It Works
1. ✅ Celebrate! You fixed class imbalance
2. Next: Add augmentations (see CODEBASE_ANALYSIS.md Priority 2)
3. Goal: Push F1 from 0.75 → 0.80+

## If It Doesn't Work
1. Check if class weights are actually loaded (print statement)
2. Try lower class loss weight (10.0 → 5.0)
3. Inspect Class_1 images manually - are labels correct?

---

**Bottom line**: 30 minutes of changes should give you +8% F1 improvement. Go implement it! 🚀
