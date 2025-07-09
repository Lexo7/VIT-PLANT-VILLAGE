"""
Unit tests for data loading and preprocessing.
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_preprocessing import PlantDiseaseDataset, get_transforms, split_dataset


class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
        
        # Create test data structure
        for split in ['train', 'val', 'test']:
            for class_name in ['healthy', 'diseased']:
                class_dir = self.data_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                for i in range(3):
                    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
                    img.save(class_dir / f"image_{i}.jpg")
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_plant_disease_dataset(self):
        """Test PlantDiseaseDataset loading."""
        transforms = get_transforms(224)
        
        dataset = PlantDiseaseDataset(
            data_dir=str(self.data_dir),
            transform=transforms['train'],
            split='train'
        )
        
        # Test dataset properties
        self.assertEqual(len(dataset), 6)  # 2 classes * 3 images each
        self.assertEqual(len(dataset.label_to_idx), 2)
        
        # Test data loading
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])
    
    def test_transforms(self):
        """Test data transforms."""
        transforms_dict = get_transforms(224)
        
        self.assertIn('train', transforms_dict)
        self.assertIn('val', transforms_dict)
        self.assertIn('test', transforms_dict)
        
        # Test transform application
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        train_transformed = transforms_dict['train'](dummy_image)
        val_transformed = transforms_dict['val'](dummy_image)
        
        self.assertEqual(train_transformed.shape, (3, 224, 224))
        self.assertEqual(val_transformed.shape, (3, 224, 224))
    
    def test_split_dataset(self):
        """Test dataset splitting functionality."""
        # Create raw data directory
        raw_dir = Path(self.temp_dir) / "raw_data"
        output_dir = Path(self.temp_dir) / "split_data"
        
        # Create test classes with images
        for class_name in ['class1', 'class2']:
            class_dir = raw_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 10 dummy images per class
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
                img.save(class_dir / f"image_{i}.jpg")
        
        # Test splitting
        split_dataset(
            str(raw_dir), 
            str(output_dir),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42
        )
        
        # Verify split directories exist
        for split in ['train', 'val', 'test']:
            self.assertTrue((output_dir / split).exists())
            
            for class_name in ['class1', 'class2']:
                self.assertTrue((output_dir / split / class_name).exists())
        
        # Verify approximate split ratios
        train_count = len(list((output_dir / 'train' / 'class1').glob('*.jpg')))
        val_count = len(list((output_dir / 'val' / 'class1').glob('*.jpg')))
        test_count = len(list((output_dir / 'test' / 'class1').glob('*.jpg')))
        
        self.assertEqual(train_count + val_count + test_count, 10)
        self.assertGreaterEqual(train_count, 5)  # Should be around 6
        self.assertGreaterEqual(val_count, 1)    # Should be around 2
        self.assertGreaterEqual(test_count, 1)   # Should be around 2


if __name__ == '__main__':
    unittest.main()
