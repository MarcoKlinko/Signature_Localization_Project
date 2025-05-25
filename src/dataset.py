# importing modules
# -*- coding: utf-8 -*-

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# creating dataset class

class SignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with 'images' and 'labels' subdirectories
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        # Get list of image files (assuming .jpg, .png, .jpeg extensions)
        self.image_files = [f for f in os.listdir(self.image_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        with open(label_path, 'r') as f:
            bbox = list(map(float, f.read().strip().split()))
        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, bbox

# Define transformations
def get_transform(train=True):
    transform_list = [
        transforms.Resize((512, 512)),  # Resize to consistent dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        # Add data augmentation only for training
        transform_list.insert(1, transforms.ColorJitter(brightness=0.2, contrast=0.2))
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    
    return transforms.Compose(transform_list)