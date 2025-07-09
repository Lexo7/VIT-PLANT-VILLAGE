"""
Data preprocessing utilities for plant disease detection dataset.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for plant disease images."""
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        """
        Args:
            data_dir: Path to the data directory
            transform: Optional transform to be applied on a sample
            split: One of 'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data()
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load image paths and corresponding labels."""
        image_paths = []
        labels = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in class_dir.glob('*.jpg'):
                    image_paths.append(str(img_file))
                    labels.append(class_name)
                    
        return image_paths, labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return image, label_idx


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Get data transforms for training and validation."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def get_albumentations_transforms(image_size: int = 224) -> Dict[str, A.Compose]:
    """Get albumentations transforms for more advanced augmentation."""
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def split_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.7, 
                  val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
    """Split dataset into train, validation, and test sets."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Process each class directory
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            random.shuffle(image_files)
            
            # Calculate split indices
            n_images = len(image_files)
            train_end = int(n_images * train_ratio)
            val_end = int(n_images * (train_ratio + val_ratio))
            
            # Split files
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # Create class directories in splits
            for split in ['train', 'val', 'test']:
                (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Copy files to respective splits
            for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
                for file in files:
                    dest = output_path / split / class_name / file.name
                    shutil.copy2(file, dest)
            
            print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4,
                       image_size: int = 224) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    
    transforms_dict = get_transforms(image_size)
    
    datasets = {}
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = PlantDiseaseDataset(
            data_dir=data_dir,
            transform=transforms_dict[split],
            split=split
        )
        
        shuffle = True if split == 'train' else False
        
        data_loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return data_loaders


def analyze_dataset(data_dir: str) -> pd.DataFrame:
    """Analyze the dataset and return statistics."""
    
    data_path = Path(data_dir)
    stats = []
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if split_path.exists():
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    n_images = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                    stats.append({
                        'split': split,
                        'class': class_name,
                        'count': n_images
                    })
    
    return pd.DataFrame(stats)


if __name__ == "__main__":
    # Example usage
    raw_data_dir = "plant_village/raw"
    processed_data_dir = "plant_village/processed"
    
    # Split dataset
    if os.path.exists(raw_data_dir):
        split_dataset(raw_data_dir, processed_data_dir)
        
        # Analyze dataset
        stats = analyze_dataset(processed_data_dir)
        print("\nDataset Statistics:")
        print(stats.pivot(index='class', columns='split', values='count').fillna(0))
    else:
        print(f"Raw data directory {raw_data_dir} not found. Please download the PlantVillage dataset.")
