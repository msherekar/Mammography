"""
Test-time augmentation (TTA) module for mammography image classification.

This module provides functions to create image transformations for both
training and inference time. Cancer-positive samples can be augmented 
differently to address class imbalance.
"""
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Callable

def get_transforms(augment: bool = False) -> Callable:
    """
    Get image transformations for training or validation.
    
    Args:
        augment: Whether to apply data augmentation (True) or just normalization (False)
        
    Returns:
        Albumentation composition of transforms
    """
    if augment:
        return albu.Compose([
            # Spatial transformations
            albu.RandomRotate90(p=1.0),
            albu.VerticalFlip(p=1.0),
            albu.HorizontalFlip(p=0.5),
            albu.Affine(rotate=(-20, 20), p=1.0),
            
            # Intensity transformations (optional)
            albu.OneOf([
                albu.RandomBrightnessContrast(p=1.0),
                albu.RandomGamma(p=1.0),
            ], p=0.3),
            
            # Convert to tensor for PyTorch
            ToTensorV2(),
        ])
    else:
        return albu.Compose([
            # Only convert to tensor for validation/testing
            ToTensorV2(),
        ])

def get_cancer_augmentations() -> Callable:
    """
    Get stronger augmentations specifically for cancer-positive samples.
    This can help with class imbalance by creating more variations of the minority class.
    
    Returns:
        Albumentation composition with stronger augmentations
    """
    return albu.Compose([
        # Spatial transformations
        albu.RandomRotate90(p=1.0),
        albu.VerticalFlip(p=1.0),
        albu.HorizontalFlip(p=0.5),
        albu.Affine(rotate=(-30, 30), scale=(0.9, 1.1), p=1.0),
        
        # Intensity transformations
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            albu.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        
        # Convert to tensor for PyTorch
        ToTensorV2(),
    ])
