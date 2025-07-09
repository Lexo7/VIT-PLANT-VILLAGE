import torch.nn as nn
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Replace the classifier
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
