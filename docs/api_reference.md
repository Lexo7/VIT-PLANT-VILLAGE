# API Reference

This document provides a reference for the main classes and functions in the plant disease detection project.

## Data Module (`data/`)

### PlantDiseaseDataset

```python
class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for plant disease images."""
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        """
        Args:
            data_dir: Path to the data directory
            transform: Optional transform to be applied on a sample
            split: One of 'train', 'val', 'test'
        """
```

### Data Preprocessing Functions

```python
def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Get data transforms for training and validation."""

def split_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.7, 
                  val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
    """Split dataset into train, validation, and test sets."""

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4,
                       image_size: int = 224) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing."""
```

## Models Module (`models/`)

### CNN Models (`models/cnn_models.py`)

#### ResNetClassifier

```python
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        """ResNet-50 based classifier for plant disease detection."""
```

#### EfficientNetClassifier

```python
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        """EfficientNet-B0 based classifier for plant disease detection."""
```

### Vision Transformer Models (`models/vit_models.py`)

#### VisionTransformer

```python
class VisionTransformer(nn.Module):
    """Vision Transformer for plant disease classification."""
    
    def __init__(self, num_classes: int, model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True):
        """
        Args:
            num_classes: Number of plant disease classes
            model_name: Pre-trained ViT model name from timm
            pretrained: Whether to use pre-trained weights
        """
```

#### CustomViT

```python
class CustomViT(nn.Module):
    """Custom Vision Transformer implementation."""
    
    def __init__(self, num_classes: int, img_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        Args:
            num_classes: Number of output classes
            img_size: Input image size
            patch_size: Size of image patches
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
```

### Hybrid Models (`models/hybrid_models.py`)

#### ParallelCNNViT

```python
class ParallelCNNViT(nn.Module):
    """Parallel CNN and ViT processing with feature fusion."""
    
    def __init__(self, num_classes: int, cnn_backbone: str = 'resnet50',
                 vit_model: str = 'vit_base_patch16_224', fusion_method: str = 'concat'):
        """
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture
            vit_model: ViT model name
            fusion_method: Method for fusing CNN and ViT features ('concat', 'add', 'multiply')
        """
```

#### AttentionFusedCNNViT

```python
class AttentionFusedCNNViT(nn.Module):
    """CNN-ViT hybrid with attention-based feature fusion."""
    
    def __init__(self, num_classes: int, cnn_backbone: str = 'resnet50',
                 vit_model: str = 'vit_base_patch16_224'):
        """
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture
            vit_model: ViT model name
        """
```

#### Model Factory Function

```python
def create_model(model_type: str, num_classes: int, **kwargs):
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Instantiated model
    """
```

## Training Module (`training/`)

### Common Training Functions

```python
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""

def validate_epoch(model, dataloader, criterion, device, epoch, writer):
    """Validate for one epoch."""

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint."""
```

## Evaluation Module (`evaluation/`)

### Model Evaluation Functions

```python
def load_model(model_type, checkpoint_path, num_classes, device, **kwargs):
    """Load a trained model from checkpoint."""

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model on a dataset."""

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix."""

def save_results(results, output_dir, model_name):
    """Save evaluation results."""
```

## Usage Examples

### Creating a Model

```python
from models.vit_models import VisionTransformer
from models.hybrid_models import create_model

# Create a Vision Transformer
vit_model = VisionTransformer(num_classes=38, model_name='vit_base_patch16_224')

# Create a hybrid model using factory function
hybrid_model = create_model(
    'parallel_cnn_vit', 
    num_classes=38,
    cnn_backbone='resnet50',
    vit_model='vit_base_patch16_224',
    fusion_method='concat'
)
```

### Loading Data

```python
from data.data_preprocessing import create_data_loaders

# Create data loaders
data_loaders = create_data_loaders(
    data_dir='data/plant_village/processed',
    batch_size=32,
    image_size=224
)

train_loader = data_loaders['train']
val_loader = data_loaders['val']
test_loader = data_loaders['test']
```

### Training a Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(num_classes=38).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch, writer
    )
    val_loss, val_acc = validate_epoch(
        model, val_loader, criterion, device, epoch, writer
    )
```

### Evaluating a Model

```python
from evaluation.evaluate_models import load_model, evaluate_model

# Load trained model
model = load_model('vit_base_patch16_224', 'path/to/checkpoint.pth', 38, device)

# Evaluate
results = evaluate_model(model, test_loader, device, class_names)
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

## Configuration

### Model Configuration

Models are configured using JSON files that store training parameters and model architecture details:

```json
{
  "model": "vit_base_patch16_224",
  "num_classes": 38,
  "batch_size": 16,
  "epochs": 50,
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "image_size": 224,
  "best_accuracy": 95.67,
  "label_to_idx": {
    "class1": 0,
    "class2": 1,
    ...
  }
}
```

### Training Arguments

All training scripts accept command-line arguments for configuration. See individual script help for details:

```bash
python training/train_vit.py --help
python training/train_cnn.py --help
python training/train_hybrid.py --help
```
