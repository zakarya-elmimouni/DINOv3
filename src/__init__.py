"""
DINOv3 Object Detection Training Module
"""
__version__ = "0.1.0"

from .dataset import YOLODetectionDataset, create_dataloaders
from .model import DINOv3Detector, create_model
from .loss import DetectionLoss, HungarianMatcher
from .utils import AverageMeter, save_checkpoint, load_checkpoint
from .train import Trainer
from .inference import Inferencer

__all__ = [
    'YOLODetectionDataset', 'create_dataloaders', 'DINOv3Detector', 
    'create_model', 'DetectionLoss', 'HungarianMatcher', 'AverageMeter',
    'save_checkpoint', 'load_checkpoint', 'Trainer', 'Inferencer'
]