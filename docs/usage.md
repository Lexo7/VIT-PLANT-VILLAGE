# Usage Guide

This guide explains how to use the plant disease detection project for training models and making predictions.

## Quick Start

### 1. Data Preparation

First, ensure your data is properly prepared:

```bash
# Split raw dataset into train/val/test
python data/data_preprocessing.py
```

### 2. Training Models

#### Train CNN Baseline Models

```bash
# Train ResNet-50
python training/train_cnn.py --model resnet50 --epochs 50 --batch_size 32

# Train EfficientNet-B0
python training/train_cnn.py --model efficientnet_b0 --epochs 50 --batch_size 32
```

#### Train Vision Transformer Models

```bash
# Train ViT-Base
python training/train_vit.py --model vit_base_patch16_224 --epochs 50 --batch_size 16

# Train Custom ViT
python training/train_vit.py --model custom_vit --epochs 50 --batch_size 16
```

#### Train Hybrid Models

```bash
# Train Parallel CNN-ViT with concatenation fusion
python training/train_hybrid.py --model parallel_cnn_vit --fusion_method concat --epochs 50

# Train Attention-fused CNN-ViT
python training/train_hybrid.py --model attention_fused_cnn_vit --epochs 50
```

### 3. Model Evaluation

```bash
# Evaluate a trained model
python evaluation/evaluate_models.py \
    --checkpoint results/model_weights/best_cnn_resnet50.pth \
    --config results/model_weights/config_cnn_resnet50.json \
    --split test
```

## Detailed Usage

### Training Parameters

All training scripts support the following common parameters:

- `--data_dir`: Path to processed dataset (default: `data/plant_village/processed`)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for regularization
- `--image_size`: Input image size (default: 224)
- `--output_dir`: Directory to save model weights
- `--log_dir`: Directory for tensorboard logs
- `--resume`: Path to checkpoint to resume training

### Model-Specific Parameters

#### CNN Models (`train_cnn.py`)
- `--model`: Choice of `resnet50` or `efficientnet_b0`

#### ViT Models (`train_vit.py`)
- `--model`: Choice of `vit_base_patch16_224`, `vit_small_patch16_224`, or `custom_vit`

#### Hybrid Models (`train_hybrid.py`)
- `--model`: Choice of `hybrid_cnn_vit`, `parallel_cnn_vit`, or `attention_fused_cnn_vit`
- `--cnn_backbone`: CNN backbone (`resnet50` or `efficientnet_b0`)
- `--vit_model`: ViT model name
- `--fusion_method`: Fusion method for parallel models (`concat`, `add`, or `multiply`)

### Example Training Commands

#### High-Performance Setup
```bash
# Train with larger batch size and longer training
python training/train_vit.py \
    --model vit_base_patch16_224 \
    --batch_size 32 \
    --epochs 100 \
    --lr 3e-4 \
    --image_size 224
```

#### Resource-Constrained Setup
```bash
# Train with smaller batch size for limited GPU memory
python training/train_cnn.py \
    --model resnet50 \
    --batch_size 8 \
    --epochs 30 \
    --lr 1e-3
```

#### Resume Training
```bash
# Resume training from a checkpoint
python training/train_vit.py \
    --model vit_base_patch16_224 \
    --resume results/model_weights/vit_base_patch16_224_epoch_20.pth
```

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir results/logs
```

Then open your browser to `http://localhost:6006` to view training metrics.

### Evaluation and Analysis

#### Basic Evaluation
```bash
python evaluation/evaluate_models.py \
    --checkpoint path/to/model.pth \
    --config path/to/config.json \
    --split test
```

#### Batch Evaluation
You can evaluate multiple models by creating a script:

```bash
#!/bin/bash
for model in results/model_weights/best_*.pth; do
    config="${model%.*}_config.json"
    python evaluation/evaluate_models.py --checkpoint "$model" --config "$config"
done
```

## Using Jupyter Notebooks

The project includes several Jupyter notebooks for interactive exploration:

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`):
   - Visualize dataset statistics
   - Explore class distributions
   - Analyze image properties

2. **Baseline Training** (`notebooks/02_baseline_training.ipynb`):
   - Train simple CNN models
   - Compare different architectures
   - Visualize training progress

3. **ViT Implementation** (`notebooks/03_vit_implementation.ipynb`):
   - Implement and train Vision Transformers
   - Visualize attention maps
   - Compare with CNN baselines

4. **Hybrid Experiments** (`notebooks/04_hybrid_experiments.ipynb`):
   - Experiment with hybrid architectures
   - Analyze fusion strategies
   - Performance comparisons

To start Jupyter:
```bash
jupyter notebook notebooks/
```

## Best Practices

### Training Tips

1. **Start Small**: Begin with smaller models and shorter training to verify your setup
2. **Monitor Overfitting**: Use validation metrics to detect overfitting early
3. **Data Augmentation**: Use appropriate augmentation for better generalization
4. **Learning Rate**: Start with recommended learning rates and adjust based on convergence
5. **Regularization**: Use weight decay and dropout to prevent overfitting

### Performance Optimization

1. **Batch Size**: Use the largest batch size that fits in your GPU memory
2. **Mixed Precision**: Enable mixed precision training for faster training on modern GPUs
3. **Data Loading**: Use multiple workers for data loading to avoid bottlenecks
4. **Checkpointing**: Save checkpoints regularly to avoid losing progress

### Debugging

1. **Small Dataset**: Test with a small subset of data first
2. **Overfit Single Batch**: Ensure your model can overfit a single batch
3. **Check Data**: Visualize your data to ensure correct preprocessing
4. **Monitor Gradients**: Check for vanishing/exploding gradients

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Check data loading bottlenecks, use GPU efficiently
3. **Poor Convergence**: Adjust learning rate, check data preprocessing
4. **Low Accuracy**: Verify data quality, try different augmentations

### Performance Issues

- **GPU Utilization**: Monitor GPU usage with `nvidia-smi`
- **CPU Bottleneck**: Increase `num_workers` in data loaders
- **Memory Usage**: Use memory profiling tools to identify leaks
