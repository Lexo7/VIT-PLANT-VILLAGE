import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import timm


class VisionTransformer(nn.Module):
    """Vision Transformer for plant disease classification."""
    
    def __init__(self, num_classes: int, model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True):
        super(VisionTransformer, self).__init__()
        
        # Load pre-trained ViT model using timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class HuggingFaceViT(nn.Module):
    """Vision Transformer using Hugging Face transformers."""
    
    def __init__(self, num_classes: int, model_name: str = 'google/vit-base-patch16-224'):
        super(HuggingFaceViT, self).__init__()
        
        # Load configuration
        config = ViTConfig.from_pretrained(model_name)
        
        # Load the model
        self.vit = ViTModel.from_pretrained(model_name, config=config)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
    def forward(self, x):
        # Pass through ViT
        outputs = self.vit(pixel_values=x)
        
        # Get the pooled output (CLS token)
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class CustomViT(nn.Module):
    """Custom Vision Transformer implementation."""
    
    def __init__(self, num_classes: int, img_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(CustomViT, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Use CLS token
        logits = self.classifier(x)
        
        return logits
