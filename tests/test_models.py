"""
Unit tests for model implementations.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_models import ResNetClassifier, EfficientNetClassifier
from models.vit_models import VisionTransformer, CustomViT
from models.hybrid_models import HybridCNNViT, ParallelCNNViT, create_model


class TestCNNModels(unittest.TestCase):
    
    def setUp(self):
        self.num_classes = 10
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
    
    def test_resnet_classifier(self):
        model = ResNetClassifier(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_efficientnet_classifier(self):
        model = EfficientNetClassifier(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestViTModels(unittest.TestCase):
    
    def setUp(self):
        self.num_classes = 10
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
    
    def test_vision_transformer(self):
        model = VisionTransformer(
            num_classes=self.num_classes,
            model_name='vit_small_patch16_224'
        )
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_custom_vit(self):
        model = CustomViT(
            num_classes=self.num_classes,
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=6,
            num_heads=6
        )
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestHybridModels(unittest.TestCase):
    
    def setUp(self):
        self.num_classes = 10
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
    
    def test_parallel_cnn_vit_concat(self):
        model = ParallelCNNViT(
            num_classes=self.num_classes,
            cnn_backbone='resnet50',
            vit_model='vit_small_patch16_224',
            fusion_method='concat'
        )
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_parallel_cnn_vit_add(self):
        model = ParallelCNNViT(
            num_classes=self.num_classes,
            cnn_backbone='resnet50',
            vit_model='vit_small_patch16_224',
            fusion_method='add'
        )
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_create_model_factory(self):
        model = create_model(
            'parallel_cnn_vit',
            num_classes=self.num_classes,
            cnn_backbone='resnet50',
            vit_model='vit_small_patch16_224',
            fusion_method='concat'
        )
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


if __name__ == '__main__':
    unittest.main()
