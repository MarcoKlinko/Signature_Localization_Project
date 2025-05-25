# importing modules
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Define the SignatureLocalizer model

class SignatureLocalizer(nn.Module):
    def __init__(self, pretrained=True):
        super(SignatureLocalizer, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Additional convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=3, padding=1),
            nn.Sigmoid()  # Outputs normalized coordinates
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        x = self.conv_layers(x)
        
        # Bounding box prediction
        bbox = self.bbox_head(x)
        
        # Global average pooling to get single prediction per image
        bbox = F.adaptive_avg_pool2d(bbox, (1, 1)).view(-1, 4)
        
        return bbox