# Advanced Transfer Learning for Amulet Classification

This directory contains scripts for training and evaluating amulet classification models using transfer learning techniques.

## Overview

The implementation uses state-of-the-art transfer learning approaches with EfficientNet and ResNet architectures to classify amulet images. The system is designed to handle the unique challenges of amulet classification, including:

- Avoiding rotation and flip augmentations which would change the meaning of amulet features
- Special handling for small classes to prevent overfitting
- Progressive unfreezing of model layers for optimal fine-tuning

## Scripts

The following scripts are included:

1. `setup.py` - Installs all required dependencies
2. `unified_dataset_creator.py` - Prepares dataset by combining and organizing images
3. `advanced_transfer_learning.py` - Main training implementation using transfer learning
4. `run_training.py` - Simplified interface to run the entire training pipeline
5. `model_evaluator.py` - Tools to evaluate trained models on test data

## Getting Started

### Installation

First, install the required dependencies:

```bash
python setup.py
```

For GPU acceleration, specify your CUDA version:

```bash
python setup.py --cuda 11.8
```

### Data Preparation

Prepare your dataset using the unified dataset creator:

```bash
python unified_dataset_creator.py --sources /path/to/data1 /path/to/data2 --output unified_dataset
```

This will organize your data into train/validation/test splits and handle class imbalance issues.

### Training

To train a model with default settings:

```bash
python run_training.py --sources /path/to/data1 /path/to/data2 --model efficientnet_b3
```

Advanced training options:

```bash
python run_training.py --sources /path/to/data1 /path/to/data2 \
    --model efficientnet_b3 \
    --batch-size 16 \
    --epochs 50 \
    --head-epochs 10 \
    --learning-rate 1e-3 \
    --output-dir training_output
```

### Evaluation

Evaluate your trained model on test data:

```bash
python model_evaluator.py --model training_output/models/finetune_best_model_metric.pth --test-dir unified_dataset/test
```

Or evaluate on a single image:

```bash
python model_evaluator.py --model training_output/models/finetune_best_model_metric.pth --image /path/to/image.jpg
```

## Model Architecture

The system supports the following architectures:

- `efficientnet_b0` - Lightweight model for faster training
- `efficientnet_b3` - Better accuracy with reasonable training time (recommended)
- `resnet50` - Alternative architecture if EfficientNet doesn't perform well

The training process follows three stages:

1. **Head-only training** - Only the classifier head is trained while the backbone is frozen
2. **Partial fine-tuning** - Last few layers of the backbone are unfrozen
3. **Full fine-tuning** - All layers are trained with a lower learning rate

## Advanced Features

- **Mixed precision training** - Accelerates training on compatible GPUs
- **Learning rate scheduling** - Warm-up followed by cosine annealing
- **Class weighting** - Handles class imbalance in the dataset
- **Early stopping** - Prevents overfitting
- **Comprehensive metrics** - Precision, recall, F1 score per class
- **Confusion matrix analysis** - Identifies problematic classes

## Results

After training, you'll find the following in your output directory:

- Trained model checkpoints
- Training history and metrics
- Visualizations (confusion matrices, learning curves)
- Comprehensive evaluation report

## Troubleshooting

If you encounter issues:

1. Check CUDA compatibility if using GPU
2. Ensure dataset is properly organized
3. Try reducing batch size if you get out-of-memory errors
4. Check training logs for specific error messages
