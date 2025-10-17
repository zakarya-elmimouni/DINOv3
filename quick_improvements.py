"""
Quick improvements to address class imbalance
Run this to implement Priority 1 changes
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_dataset_distribution(labels_dir):
    """Analyze class distribution in dataset"""
    labels_dir = Path(labels_dir)
    class_counts = Counter()
    total_instances = 0
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_instances += 1
    
    num_classes = max(class_counts.keys()) + 1
    
    print(f"\n{'='*60}")
    print(f"Dataset Class Distribution")
    print(f"{'='*60}")
    
    class_dist = {}
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        percentage = (count / total_instances) * 100
        print(f"Class_{i}: {int(count):5d} instances ({percentage:5.2f}%)")
        class_dist[i] = count
    
    print(f"{'='*60}")
    print(f"Total: {total_instances} instances across {num_classes} classes")
    print(f"{'='*60}\n")
    
    return class_dist, num_classes

def compute_class_weights(class_dist, num_classes, method='inverse_freq'):
    """
    Compute class weights for loss function
    
    Methods:
    - 'inverse_freq': weight = 1 / frequency
    - 'balanced': weight = total / (num_classes * count)
    - 'sqrt_inv_freq': weight = 1 / sqrt(frequency)
    """
    total = sum(class_dist.values())
    weights = []
    
    print(f"\n{'='*60}")
    print(f"Computing Class Weights (method: {method})")
    print(f"{'='*60}")
    
    for i in range(num_classes):
        count = class_dist.get(i, 1)  # Avoid division by zero
        
        if method == 'inverse_freq':
            weight = total / (count * num_classes)
        elif method == 'balanced':
            weight = total / (num_classes * count)
        elif method == 'sqrt_inv_freq':
            weight = np.sqrt(total / count)
        else:
            weight = 1.0
        
        weights.append(weight)
        print(f"Class_{i}: weight = {weight:.4f} (count = {count})")
    
    # Add background class weight (usually 1.0)
    weights.append(1.0)
    print(f"Background: weight = 1.0000")
    print(f"{'='*60}\n")
    
    return weights

def create_sample_weights_file(train_labels_dir, val_labels_dir, output_file='dataset/sample_weights.json'):
    """Create sample weights for balanced sampling"""
    print("Analyzing training set...")
    train_dist, num_classes = analyze_dataset_distribution(train_labels_dir)
    
    print("Analyzing validation set...")
    val_dist, _ = analyze_dataset_distribution(val_labels_dir)
    
    # Compute class weights based on training set
    class_weights = compute_class_weights(train_dist, num_classes, method='inverse_freq')
    
    # Create sample weights for each training image
    print("Creating sample weights for balanced sampling...")
    train_labels_dir = Path(train_labels_dir)
    sample_weights = []
    
    for label_file in sorted(train_labels_dir.glob("*.txt")):
        # Get classes present in this image
        image_classes = set()
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    image_classes.add(class_id)
        
        # Weight = max weight of any class in image (prioritize rare classes)
        if image_classes:
            max_weight = max(class_weights[c] for c in image_classes)
        else:
            max_weight = 1.0
        
        sample_weights.append(float(max_weight))
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    results = {
        'num_classes': num_classes,
        'class_distribution': {f"class_{i}": count for i, count in train_dist.items()},
        'class_weights': [float(w) for w in class_weights],
        'sample_weights': sample_weights,
        'method': 'inverse_freq',
        'description': 'Class weights for loss function and sample weights for balanced sampling'
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved weights to: {output_path}")
    print(f"   - Class weights: {len(class_weights)} values")
    print(f"   - Sample weights: {len(sample_weights)} values")
    
    # Print recommendations
    print(f"\n{'='*60}")
    print("📋 Implementation Recommendations")
    print(f"{'='*60}")
    print("\n1. Update config.yaml:")
    print("   loss_weights:")
    print("     class: 10.0  # Increased from 1.0")
    print("     bbox: 5.0")
    print("     giou: 2.0")
    print("     obj: 1.0")
    
    print("\n2. In loss.py, add class_weights parameter:")
    print("   criterion = DetectionLoss(")
    print("       num_classes=num_classes,")
    print(f"       class_weights={class_weights[:num_classes]},")
    print("       loss_weights=config['loss_weights']")
    print("   )")
    
    print("\n3. In dataset.py, use WeightedRandomSampler:")
    print("   from torch.utils.data import WeightedRandomSampler")
    print("   sampler = WeightedRandomSampler(")
    print("       weights=sample_weights,")
    print("       num_samples=len(sample_weights),")
    print("       replacement=True")
    print("   )")
    print("   train_loader = DataLoader(..., sampler=sampler, shuffle=False)")
    print(f"{'='*60}\n")
    
    return results

def analyze_confusion_matrix(results_file='checkpoints-v1/results.json'):
    """Analyze confusion matrix to identify problem classes"""
    print(f"\n{'='*60}")
    print("Confusion Matrix Analysis")
    print(f"{'='*60}\n")
    
    # This would load and analyze the confusion matrix from your results
    # For now, using your provided numbers:
    
    confusion_data = {
        0: {'correct': 717, 'misclassified': {}},
        1: {'correct': 26, 'misclassified': {0: 161, 2: 223}},
        2: {'correct': 233, 'misclassified': {0: 226}}
    }
    
    for class_id, data in confusion_data.items():
        total = data['correct'] + sum(data['misclassified'].values())
        accuracy = data['correct'] / total if total > 0 else 0
        
        print(f"Class_{class_id}:")
        print(f"  Correct: {data['correct']:4d} / {total:4d} ({accuracy*100:5.2f}%)")
        
        if data['misclassified']:
            print(f"  Misclassified as:")
            for wrong_class, count in data['misclassified'].items():
                pct = count / total * 100
                print(f"    Class_{wrong_class}: {count:3d} ({pct:5.2f}%)")
        print()
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset and compute class weights")
    parser.add_argument('--train_labels', default='dataset/train/labels', help='Path to training labels')
    parser.add_argument('--val_labels', default='dataset/val/labels', help='Path to validation labels')
    parser.add_argument('--output', default='dataset/sample_weights.json', help='Output file for weights')
    args = parser.parse_args()
    
    print("="*60)
    print("Quick Improvements for Class Imbalance")
    print("="*60)
    
    # Analyze confusion matrix from previous run
    analyze_confusion_matrix()
    
    # Compute and save weights
    results = create_sample_weights_file(args.train_labels, args.val_labels, args.output)
    
    print("\n✅ Analysis complete!")
    print(f"   Next steps: Follow the implementation recommendations above")
    print(f"   Expected improvement: F1 0.693 → 0.75+ (+8%)")
