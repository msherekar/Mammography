"""
Main entry point for the mammography AI training and evaluation.

This script handles command-line arguments and launches the appropriate
training or evaluation functions.
"""
import os
import argparse
import datetime
import json
import shutil
from typing import Dict, Any

import torch
import mmengine
from mmcls.utils import register_all_modules

# Import project modules
from constants import INPUT_CHK_PT, THRESHOLD, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS
from training import train_model, train_model_loss_attenuation

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Mammography AI model training')
    
    # Experiment identification
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    
    # Data paths
    data_group = parser.add_argument_group('Data Paths')
    data_group.add_argument('--train_csv_pth', type=str, required=True, help='Path to training CSV file')
    data_group.add_argument('--valid_csv_pth', type=str, required=True, help='Path to validation CSV file')
    data_group.add_argument('--dcm_root_dir_pth', type=str, help='Root directory of DICOM images')
    data_group.add_argument('--save_img_root_dir_pth', type=str, help='Directory to save processed PNG images')
    data_group.add_argument('--out_dir', type=str, required=True, help='Output directory for results')
    
    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset Options')
    dataset_group.add_argument('--data', type=str, default='EMBED', choices=['RSNA', 'EMBED'], 
                              help='Choose dataset type')
    dataset_group.add_argument('--remove_processed_pngs', type=str, default='False', 
                              help='Whether to remove generated PNGs after training')
    dataset_group.add_argument('--dataset_partition', action='store_true', default=False,
                              help='Use a subset of data for experimentation')
    dataset_group.add_argument('--rows_experiment', type=int, default=5000,
                              help='Number of rows to use when dataset_partition is True')
    dataset_group.add_argument('--finite_sample_rate', type=float, 
                              help='Rate for finite sampling if needed')
    dataset_group.add_argument('--sampler', type=str, default='Weighted', 
                              choices=['Weighted', 'Balanced', 'Simple'],
                              help='Sampling strategy to use')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--fine_tuning', type=str, default='full', choices=['full', 'partial'],
                            help='Type of fine-tuning to apply')
    model_group.add_argument('--upto_freeze', type=int, default=0, 
                            help='Number of layers to freeze (max 42)')
    model_group.add_argument('--dropout_rate', type=float,
                            help='Dropout rate to apply to the model')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    train_group.add_argument('--num_epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    train_group.add_argument('--start_learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                            help='Initial learning rate')
    train_group.add_argument('--optimizer', type=str, default='adamW', 
                           choices=['adam', 'sgd', 'adamW'], help='Optimizer to use')
    train_group.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    train_group.add_argument('--training_augment', action='store_true', default=False,
                            help='Apply data augmentation during training')
    train_group.add_argument('--loss_function', type=str, default='BCELogitsLoss',
                           choices=['FocalLoss', 'BCELoss', 'BCELogitsLoss', 'Softmax'],
                           help='Loss function to use')
    train_group.add_argument('--learned_loss_attnuation', action='store_true', default=False,
                            help='Apply learned loss attenuation')
    train_group.add_argument('--t_number', type=int, default=10,
                            help='Number of Monte Carlo samples for loss attenuation')
    
    # Learning rate schedule
    lr_group = parser.add_argument_group('Learning Rate Schedule')
    lr_group.add_argument('--decay_every_N_epoch', type=int, default=5,
                         help='Decay learning rate every N epochs')
    lr_group.add_argument('--decay_multiplier', type=float, default=0.95,
                         help='Multiplier for learning rate decay')
    
    # Checkpointing and logging
    logging_group = parser.add_argument_group('Checkpointing and Logging')
    logging_group.add_argument('--save_every_N_epochs', type=int, default=1,
                              help='Save checkpoint every N epochs')
    logging_group.add_argument('--bsave_valid_results_at_epochs', action='store_true', default=False,
                              help='Save validation results at every epoch')
    
    # System options
    system_group = parser.add_argument_group('System Options')
    system_group.add_argument('--threads', type=int, default=4, help='Number of worker threads')
    system_group.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch factor for data loading')
    system_group.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def train(args: argparse.Namespace, exp_name: str) -> None:
    """
    Main training function that dispatches to appropriate training implementation.
    
    Args:
        args: Command line arguments
        exp_name: Experiment name with timestamp
    """
    # Set random seeds for reproducibility
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic = True
    
    # Log start of training
    print(f"Starting training experiment: {exp_name}")
    print(f"Training with {args.num_epochs} epochs, batch size {args.batch_size}, "
          f"learning rate {args.start_learning_rate}")
    
    # Choose appropriate training function
    if args.learned_loss_attnuation:
        print('Training with learned loss attenuation...')
        train_model_loss_attenuation(args, INPUT_CHK_PT, THRESHOLD, exp_name)
    else:
        print('Training with standard approach...')
        train_model(args, INPUT_CHK_PT, THRESHOLD, exp_name)

    # Clean up temporary files if requested
    if args.remove_processed_pngs.lower() == 'true':
        print(f'WARNING: Deleting {args.save_img_root_dir_pth}')
        shutil.rmtree(args.save_img_root_dir_pth)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    # Register MMClassification modules
    register_all_modules()
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.jit.enable_onednn_fusion(True)
    torch.cuda.empty_cache()
    
    # Parse arguments and prepare experiment name
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.exp_name}__{timestamp}"
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save arguments for reproducibility
    args_log_file = os.path.join(args.out_dir, exp_name)
    print(f"Saving experiment configuration to: {args_log_file}")
    with open(args_log_file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    
    # Start training
    train(args, exp_name)
    print('Training completed successfully!')