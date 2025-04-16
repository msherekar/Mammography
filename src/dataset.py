import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, Any, Callable

class BreastDataset(Dataset):
    """
    Dataset class for loading breast mammography images.
    
    Supports selective augmentation for positive cancer samples to address class imbalance.
    
    Attributes:
        paths: Array of image file paths
        labels: Binary labels (1 for cancer, 0 for non-cancer)
        datasettype: Type of dataset (e.g., 'csaw', 'embed')
        transforms: Default transformations applied to all images
        augment_transforms: Special augmentations applied only to cancer-positive images
    """
    def __init__(self, 
                 df: pd.DataFrame, 
                 transforms: Optional[Callable] = None, 
                 augment_transforms: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing image paths and labels
            transforms: Default transformations for all images
            augment_transforms: Special augmentations for cancer-positive samples
        """
        self.paths = df['path'].values
        self.labels = df['cancer'].values
        self.datasettype = df['datasettype'].values
        self.transforms = transforms
        self.augment_transforms = augment_transforms

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample to get
            
        Returns:
            tuple: (image, label, dataset_type, index)
            
        Raises:
            ValueError: If there's an error loading or processing the image
        """
        # Load image
        image_path = self.paths[index]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image at path: {image_path}")
            
            # Convert BGR (default by cv2) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Retrieve label and dataset type
            label = self.labels[index]
            datasettype = self.datasettype[index]

            # Apply augmentation transforms for cancer-positive images
            if label == 1 and self.augment_transforms:
                image = self.augment_transforms(image=image)["image"]
            elif self.transforms:
                image = self.transforms(image=image)["image"]

        except Exception as e:
            # Log the error and re-raise
            print(f"Error processing image at {image_path}: {str(e)}")
            raise ValueError(f"Error loading or processing image: {str(e)}")

        return image, label, datasettype, index
