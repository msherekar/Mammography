import albumentations as albu
from albumentations.pytorch import ToTensorV2

def get_transforms(augment=False):
    if augment:
        return albu.Compose([
            albu.RandomRotate90(p=1.0),  # Random 90-degree rotations
            albu.VerticalFlip(p=1.0),   # Vertical flips
            albu.Affine(rotate=(-20, 20), p=1.0),  # Random rotations between -20 and 20 degrees
            ToTensorV2(),               # Convert to PyTorch tensor
        ])
    else:
        return albu.Compose([
            ToTensorV2(),               # For validation, just convert to tensor
        ])
