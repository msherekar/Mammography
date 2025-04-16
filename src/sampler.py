"""
Custom samplers for handling imbalanced datasets in mammography classification.

This module provides samplers that can be used with PyTorch DataLoader to ensure
balanced batches or weighted sampling strategies for dealing with class imbalance.
"""
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import Iterator, List, Optional, Tuple, Union, Sequence

class BalancedBatchSampler(Sampler):
    """
    Custom sampler that ensures each batch has an equal number of positive and negative samples.
    
    This is useful for highly imbalanced datasets where one class (typically negative) 
    significantly outnumbers the other class (positive).
    """
    def __init__(self, labels: Sequence[int], batch_size: int):
        """
        Initialize the balanced batch sampler.
        
        Args:
            labels: List of class labels for each sample (0 for negative, 1 for positive)
            batch_size: Desired batch size, must be even for equal sampling
            
        Raises:
            AssertionError: If batch_size is not even
        """
        assert batch_size % 2 == 0, "Batch size must be even to balance positive and negative samples."
        
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        
        # Calculate number of complete batches we can create
        samples_per_class = batch_size // 2
        self.num_batches = min(
            len(self.positive_indices) // samples_per_class,
            len(self.negative_indices) // samples_per_class
        )
        
        print(f"BalancedBatchSampler initialized with {len(self.positive_indices)} positive and "
              f"{len(self.negative_indices)} negative samples")
        print(f"Creating {self.num_batches} batches with {batch_size} samples each "
              f"({samples_per_class} from each class)")
    
    def __len__(self) -> int:
        """Return the number of batches that will be created."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate balanced batches by sampling equally from positive and negative classes.
        
        Returns:
            Iterator over batches, where each batch is a list of indices
        """
        # Shuffle indices for each epoch
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)
        
        # Determine samples per class
        samples_per_class = self.batch_size // 2
        
        # Create batches
        for i in range(self.num_batches):
            # Get positive samples for this batch
            pos_indices = self.positive_indices[i * samples_per_class:(i + 1) * samples_per_class]
            
            # Get negative samples for this batch
            neg_indices = self.negative_indices[i * samples_per_class:(i + 1) * samples_per_class]
            
            # Combine and shuffle
            batch_indices = np.concatenate((pos_indices, neg_indices))
            np.random.shuffle(batch_indices)
            
            yield batch_indices.tolist()


class ImbalancedSampler(WeightedRandomSampler):
    """
    A sampler that handles imbalanced datasets by assigning weights inversely proportional 
    to class frequencies.
    """
    def __init__(self, labels: Sequence[int], replacement: bool = True):
        """
        Initialize the imbalanced sampler.
        
        Args:
            labels: Class labels for the dataset
            replacement: Whether to sample with replacement
        """
        # Convert to numpy array if not already
        labels_array = np.array(labels)
        
        # Count class frequencies
        class_counts = np.bincount(labels_array)
        
        # Compute weights (inverse of frequency)
        class_weights = 1.0 / class_counts
        
        # Normalize weights so they sum to 1
        class_weights = class_weights / np.sum(class_weights)
        
        # Assign weights to each sample based on its class
        sample_weights = np.array([class_weights[label] for label in labels_array])
        
        # Initialize the parent WeightedRandomSampler
        super().__init__(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(labels_array),
            replacement=replacement
        )
        
        # Log information
        print(f"ImbalancedSampler initialized with class distribution: {class_counts}")
        print(f"Class weights: {class_weights}")


def compute_class_weights(labels: Sequence[int]) -> np.ndarray:
    """
    Compute sample weights inversely proportional to class frequencies.
    
    Args:
        labels: Class labels for all samples
        
    Returns:
        Array of weights for each sample
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Compute weights that are inversely proportional to class frequencies
    weights_per_class = total_samples / (len(class_counts) * class_counts)
    
    # Assign weights to samples
    sample_weights = np.array([weights_per_class[label] for label in labels])
    
    return sample_weights
    
