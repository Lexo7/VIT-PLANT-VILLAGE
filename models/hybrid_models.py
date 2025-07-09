import torch
import torch.nn as nn
import torchvision.models as models
import timm


class HybridCNNViT(nn.Module):
    """Hybrid model combining CNN feature extraction with Vision Transformer."""
    
    def __init__(self, num_classes: int, cnn_backbone: str = 'resnet50',
                 vit_model: str = 'vit_base_patch16_224'):
        super(HybridCNNViT, self).__init__()
        
        # CNN backbone for feature extraction
        if cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
            cnn_output_channels = 2048
        elif cnn_backbone == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=True).features
            cnn_output_channels = 1280
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # Vision Transformer for sequence modeling
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
        vit_features = self.vit.num_features
        
        # Adaptive pooling to reduce CNN feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # 14x14 = 196 patches
        
        # Feature projection to match ViT input dimensions
        self.feature_projection = nn.Linear(cnn_output_channels, vit_features)
        
        # Classification head
        self.classifier = nn.Linear(vit_features, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn(x)  # [B, C, H, W]
        
        # Adaptive pooling
        cnn_features = self.adaptive_pool(cnn_features)  # [B, C, 14, 14]
        
        # Reshape for ViT input
        B, C, H, W = cnn_features.shape
        cnn_features = cnn_features.view(B, C, H * W).transpose(1, 2)  # [B, 196, C]
        
        # Project to ViT dimensions
        projected_features = self.feature_projection(cnn_features)  # [B, 196, vit_features]
        
        # Vision Transformer processing
        vit_features = self.vit.forward_features(projected_features)
        
        # Classification
        logits = self.classifier(vit_features)
        
        return logits


class ParallelCNNViT(nn.Module):
    """Parallel CNN and ViT processing with feature fusion."""
    
    def __init__(self, num_classes: int, cnn_backbone: str = 'resnet50',
                 vit_model: str = 'vit_base_patch16_224', fusion_method: str = 'concat'):
        super(ParallelCNNViT, self).__init__()
        
        self.fusion_method = fusion_method
        
        # CNN branch
        if cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
            self.cnn.fc = nn.Identity()  # Remove final layer
            cnn_features = 2048
        elif cnn_backbone == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=True)
            self.cnn.classifier = nn.Identity()  # Remove final layer
            cnn_features = 1280
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # ViT branch
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
        vit_features = self.vit.num_features
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_dim = cnn_features + vit_features
        elif fusion_method == 'add':
            # Project CNN features to match ViT features
            self.cnn_projection = nn.Linear(cnn_features, vit_features)
            fusion_input_dim = vit_features
        elif fusion_method == 'multiply':
            # Project CNN features to match ViT features
            self.cnn_projection = nn.Linear(cnn_features, vit_features)
            fusion_input_dim = vit_features
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # CNN branch
        cnn_features = self.cnn(x)
        
        # ViT branch
        vit_features = self.vit(x)
        
        # Feature fusion
        if self.fusion_method == 'concat':
            fused_features = torch.cat([cnn_features, vit_features], dim=1)
        elif self.fusion_method == 'add':
            cnn_projected = self.cnn_projection(cnn_features)
            fused_features = cnn_projected + vit_features
        elif self.fusion_method == 'multiply':
            cnn_projected = self.cnn_projection(cnn_features)
            fused_features = cnn_projected * vit_features
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


class AttentionFusion(nn.Module):
    """Attention-based fusion of CNN and ViT features."""
    
    def __init__(self, cnn_features: int, vit_features: int, hidden_dim: int = 256):
        super(AttentionFusion, self).__init__()
        
        self.cnn_projection = nn.Linear(cnn_features, hidden_dim)
        self.vit_projection = nn.Linear(vit_features, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, cnn_features, vit_features):
        # Project features to common dimension
        cnn_proj = self.cnn_projection(cnn_features).unsqueeze(1)  # [B, 1, hidden_dim]
        vit_proj = self.vit_projection(vit_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Concatenate for attention
        features = torch.cat([cnn_proj, vit_proj], dim=1)  # [B, 2, hidden_dim]
        
        # Apply attention
        attended_features, _ = self.attention(features, features, features)
        
        # Pool attended features
        fused_features = attended_features.mean(dim=1)  # [B, hidden_dim]
        
        return fused_features


class AttentionFusedCNNViT(nn.Module):
    """CNN-ViT hybrid with attention-based feature fusion."""
    
    def __init__(self, num_classes: int, cnn_backbone: str = 'resnet50',
                 vit_model: str = 'vit_base_patch16_224'):
        super(AttentionFusedCNNViT, self).__init__()
        
        # CNN branch
        if cnn_backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
            self.cnn.fc = nn.Identity()
            cnn_features = 2048
        elif cnn_backbone == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=True)
            self.cnn.classifier = nn.Identity()
            cnn_features = 1280
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # ViT branch
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
        vit_features = self.vit.num_features
        
        # Attention fusion
        self.fusion = AttentionFusion(cnn_features, vit_features)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # CNN branch
        cnn_features = self.cnn(x)
        
        # ViT branch
        vit_features = self.vit(x)
        
        # Attention fusion
        fused_features = self.fusion(cnn_features, vit_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


def create_model(model_type: str, num_classes: int, **kwargs):
    """Factory function to create models."""
    
    if model_type == 'hybrid_cnn_vit':
        return HybridCNNViT(num_classes, **kwargs)
    elif model_type == 'parallel_cnn_vit':
        return ParallelCNNViT(num_classes, **kwargs)
    elif model_type == 'attention_fused_cnn_vit':
        return AttentionFusedCNNViT(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
