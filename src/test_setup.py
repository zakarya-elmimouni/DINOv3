"""
Test script to verify project setup and imports.
Run from project root: python -m src.test_setup
"""
import sys
from pathlib import Path
from tqdm import tqdm

def test_imports():
    print("="*60, "\nTesting Module Imports\n" + "="*60)
    results = {}
    modules = ['torch', 'torchvision', 'albumentations', 'cv2', 'scipy', 'tqdm', 
               'src.dataset', 'src.model', 'src.loss', 'src.utils', 'src.train', 'src.inference']
    for name in modules:
        try:
            mod = __import__(name, fromlist=['__version__'])
            ver = getattr(mod, '__version__', 'N/A')
            print(f"✓ {name} (v{ver}) imported.")
            if name == 'torch' and torch.cuda.is_available(): print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            results[name] = True
        except Exception as e:
            print(f"✗ {name} import failed: {e}")
            results[name] = False
    return all(results.values())

def test_dataset_structure():
    print("\n" + "="*60, "\nTesting Dataset Structure\n" + "="*60)
    dirs = ['dataset/train/images', 'dataset/train/labels', 'dataset/val/images', 'dataset/val/labels', 'dataset/test/images']
    ok = True
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            ext = '*.txt' if 'labels' in d else '*.jpg'
            count = len(list(p.glob(ext)))
            print(f"✓ {d} exists ({count} files)")
        else:
            print(f"✗ {d} does NOT exist.")
            ok = False
    return ok

def test_dataset_loading():
    print("\n" + "="*60, "\nTesting Dataset Loading\n" + "="*60)
    try:
        from src.dataset import create_dataloaders
        train_loader, _, _ = create_dataloaders('dataset/train/images', 'dataset/train/labels', 'dataset/val/images', 'dataset/val/labels', 2, 224, 0)
        images, _, _ = next(iter(train_loader))
        print(f"✓ Sample batch loaded: images shape {images.shape}")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_model_creation():
    print("\n" + "="*60, "\nTesting Model Creation\n" + "="*60)
    try:
        import torch
        from src.model import create_model
        
        safetensors_path = "weights/dinov3-vitl16-pretrain-lvd1689m/dinov3-vitl16-pretrain-lvd1689m.safetensors"

        print(f"Creating model and loading weights from: {safetensors_path}")
        model = create_model(
            num_classes=10, 
            backbone='dinov3_vitl16',  # Ensure this matches your weights file
            backbone_weights_path=safetensors_path
        )
        
        # Use a dummy input size that is a multiple of the patch size (16 for vitl16)
        dummy_input = torch.randn(1, 3, 224, 224) 
        
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)

        print("✓ Model created and forward pass successful.")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60, "\nDINOv3 Object Detection - Setup Verification\n" + "="*60)
    results = {"Imports": test_imports(), "Dataset Structure": test_dataset_structure()}
    results["Dataset Loading"] = test_dataset_loading() if results["Dataset Structure"] else False
    results["Model Creation"] = test_model_creation()
    
    print("\n" + "="*60, "\nTest Summary\n" + "="*60)
    all_passed = all(results.values())
    for name, passed in results.items(): print(f"{name}: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    print("\n" + "="*60)
    if all_passed: print("✅ All tests PASSED! Setup is complete.")
    else: print("❌ Some tests FAILED. Please fix the issues above.")
    return 0 if all_passed else 1

if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    sys.exit(main())