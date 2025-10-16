"""
Utility functions for training and evaluation
"""
import torch
import shutil
from pathlib import Path

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0: self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves checkpoint, and if it's the best, copies it to best_model.pth."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = checkpoint_dir / 'last_checkpoint.pth'
    torch.save(state, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        shutil.copyfile(checkpoint_path, best_path)
        print(f"✓ Best model saved with val_loss: {state['best_val_loss']:.4f}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Loads checkpoint from a file."""
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return 0, float('inf'), {}
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
    
    print(f"✓ Loaded from epoch {epoch} with best val loss {best_val_loss:.4f}")
    
    return epoch, best_val_loss, history