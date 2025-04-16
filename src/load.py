import os
import pandas as pd
from typing import Optional, Any

from dataset import BreastDataset
from tta import get_transforms
from finite_sample import data_sampling

def load_data_from_csv(args: Any, csv_pth: str, training_flag: bool = False) -> BreastDataset:
    """
    Load data from CSV and create appropriate dataset based on arguments.
    
    Args:
        args: Command line arguments
        csv_pth: Path to the CSV file containing dataset information
        training_flag: Whether this is for training (enables augmentation)
        
    Returns:
        BreastDataset: The dataset object ready for use with DataLoader
        
    Raises:
        RuntimeError: If unknown data type is specified
        FileNotFoundError: If CSV file doesn't exist
    """
    # Check if file exists
    if not os.path.exists(csv_pth):
        raise FileNotFoundError(f"CSV file not found at {csv_pth}")
    
    # Load data
    if training_flag and args.dataset_partition:
        df = pd.read_csv(csv_pth, nrows=args.rows_experiment)
        print(f'Loading {args.rows_experiment} rows for experimentation')
        print(f'DataFrame shape: {df.shape}')
    else:
        print(f'Loading the whole dataset for a full final run')
        df = pd.read_csv(csv_pth)
        print(f'DataFrame shape: {df.shape}')
    
    # Apply finite sampling if specified
    if training_flag and (args.finite_sample_rate is not None):
        print(f'Applying finite sampling with rate {args.finite_sample_rate}')
        df = data_sampling(df, args.finite_sample_rate)
        
        # Save sampled dataset for reference
        output_path = os.path.join(args.out_dir, f'training_list_sample_rate_{args.finite_sample_rate}.csv')
        df.to_csv(output_path, index=False)
        print(f'Saved sampled dataset to {output_path}')
    
    # Set path based on dataset type
    if args.data == 'EMBED':
        df['path'] = df["path_eq"]
    else:
        raise RuntimeError(f'Unknown data type: {args.data}. Must be EMBED.')
    
    # Create dataset with appropriate transforms
    if training_flag and args.training_augment:
        dataset = BreastDataset(
            df=df,
            transforms=get_transforms(augment=False),
            augment_transforms=get_transforms(augment=True)
        )
        print("Training with selective augmentation for cancer-positive images")
    else:
        dataset = BreastDataset(
            df=df,
            transforms=get_transforms(augment=False)
        )
    
    return dataset