"""
Script to download the correct DINOv3 weights
"""
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

if __name__ == '__main__':
    # Official DINOv3 ViT-L/16 weights URL
    url = "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidmg1Z2Ezbm43NmpzaWRic2VsN3V1YzB0IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjA4MDk4MjF9fX1dfQ__&Signature=sYHyOYELyv6VcNgpZDUA-GmjxltCULQZ8qvEnx-DT3vYlC5Gi9PT4mIKB4c9X3Wvj0HVduQ0tHUGQRCcz2QDDgYVkmG%7EAP2PtBY45otuwxKk2TQek1HE1fFASSCIIbzGWJScR0DKseeHAqQyJOc2RgzVoS75IDpPFABjAwAPUmmF1SbBJgLZHcrqpI08C6RyLZops1aLcC8515Xc4jFhigpvsGqTb03yqlcxCXN1iThQsxH2hW3SAuZ-cVm8sZALr%7E9NMXr2DVOe7iUMn40Nh7R7MGXO5Kjh6ThZA-Dpbv00FacAYOWYRJfcIxeUVbPq85nQLHiFNPfzuFTBvncbsA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1819297225392060"
    destination = "weights/dinov3-vitl16-pretrain-lvd1689m.pth"
    
    print(f"Downloading DINOv3 ViT-L/16 weights...")
    print(f"From: {url}")
    print(f"To: {destination}\n")
    
    try:
        download_file(url, destination)
        print(f"\n✓ Download complete!")
        print(f"  Saved to: {destination}")
        print(f"\nNow update your config.yaml:")
        print(f"  backbone_weights_path: '{destination}'")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print(f"\nAlternative: You can manually download from:")
        print(f"  {url}")
        print(f"  And save it to: {destination}")
