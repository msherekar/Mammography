# MammographyAI: Deep Learning for Breast Cancer Detection

A robust deep learning framework for breast cancer detection in mammography images, addressing class imbalance and model uncertainty.

## Project Overview

MammographyAI is a comprehensive deep learning solution for automated breast cancer detection in mammography images. The system leverages transfer learning with ConvNeXtV1 architecture to provide accurate cancer classification while accounting for uncertainty in predictions.

### Key Features

- **Advanced CNN Architecture**: Utilizes ConvNeXtV1 for feature extraction
- **Class Imbalance Handling**: Implements weighted sampling and selective augmentation for cancer-positive samples
- **Uncertainty Quantification**: Provides uncertainty estimates with learned loss attenuation
- **Performance Evaluation**: Comprehensive metrics including ROC-AUC, sensitivity, and specificity analysis
- **Dataset Integration**: Supports multiple mammography datasets (EMBED, CSAW)

## Architecture

The system employs a multi-component architecture:

1. **Data Processing Pipeline**: 
   - Image loading and preprocessing
   - Class-aware augmentation strategies
   - Balanced sampling for handling class imbalance

2. **Model Architecture**:
   - ConvNeXtV1 backbone with transfer learning
   - Custom classification head for binary classification
   - Optional aleatoric uncertainty estimation

3. **Training Strategies**:
   - Learning rate scheduling
   - Multiple optimizer options (Adam, SGD, AdamW)
   - Customizable loss functions (BCE, Focal Loss, etc.)

4. **Evaluation Components**:
   - Comprehensive metrics visualization
   - Dataset-specific performance analysis
   - Uncertainty visualization and analysis

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MammographyAI.git
   cd MammographyAI
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download pretrained model checkpoint:
   ```bash
   mkdir -p CHECKPOINTS
   # Download checkpoint to CHECKPOINTS directory
   ```

## Usage

### Data Preparation

Place your mammography datasets in the appropriate directory structure:

```
DATA/
├── train.csv  # Training data metadata
├── val.csv    # Validation data metadata
└── images/    # Image directory
```


### Training

To train the model with default parameters:

```bash
python src/main.py \
  --exp_name experiment_name \
  --train_csv_pth DATA/train.csv \
  --valid_csv_pth DATA/val.csv \
  --data EMBED \
  --out_dir OUT/experiment_name \
  --batch_size 32 \
  --num_epochs 10 \
  --start_learning_rate 1e-6 \
  --optimizer adamW \
  --training_augment
```

For uncertainty quantification with learned loss attenuation:

```bash
python src/main.py \
  --exp_name uncertainty_experiment \
  --train_csv_pth DATA/train.csv \
  --valid_csv_pth DATA/val.csv \
  --data EMBED \
  --out_dir OUT/uncertainty_experiment \
  --learned_loss_attnuation \
  --t_number 10
```

### Hyperparameter Tuning

The framework supports extensive hyperparameter configuration:

```bash
python src/main.py \
  --exp_name tuning_experiment \
  --train_csv_pth DATA/train.csv \
  --valid_csv_pth DATA/val.csv \
  --data EMBED \
  --out_dir OUT/tuning_experiment \
  --batch_size 16 \
  --num_epochs 20 \
  --start_learning_rate 5e-7 \
  --optimizer adamW \
  --weight_decay 1e-5 \
  --dropout_rate 0.2 \
  --loss_function FocalLoss \
  --sampler Balanced
```

## Results Visualization

After training, various plots are generated in the output directory:

- ROC curves
- Loss and AUC progression
- Sensitivity and specificity metrics
- Score distribution plots
- Uncertainty analysis plots

## Advanced Features

### Class Imbalance Handling

The system offers multiple strategies for handling class imbalance:

1. **Weighted Sampling**: Assigns higher sampling weight to cancer-positive samples
2. **Balanced Batch Sampling**: Ensures each batch has equal representation of classes
3. **Selective Augmentation**: Applies stronger augmentation to cancer-positive samples

### Transfer Learning Options

Configure transfer learning behavior:

```bash
python src/main.py \
  --exp_name transfer_learning \
  --fine_tuning partial \
  --upto_freeze 24 \
  # other parameters...
```

### Subset Experimentation

For quick experimentation on a subset of data:

```bash
python src/main.py \
  --exp_name quick_test \
  --dataset_partition \
  --rows_experiment 1000 \
  # other parameters...
```

### Code Structure

```
src/
├── config/                 # Configuration files
│   ├── __init__.py         
│   └── constants.py        # Constants and configuration
├── data/                   # Data loading and processing 
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   ├── transforms.py       # Image transformations
│   └── samplers.py         # Sampling strategies
├── models/                 # Model definitions
│   ├── __init__.py
│   └── model_utils.py      # Model building and utilities
├── training/               # Training logic
│   ├── __init__.py
│   ├── trainer.py          # Main training loops
│   └── metrics.py          # Evaluation metrics
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py    # Image preprocessing
│   ├── plotting.py         # Visualization functions
│   └── file_handling.py    # File operations
└── main.py                 # Entry point
```








