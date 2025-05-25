# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import SignatureLocalizer

# Predicting signature bounding box from an image using a trained model
def predict_signature(model_path, image_path, output_path=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SignatureLocalizer(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    original_h, original_w = original_image.shape[:2]
    input_image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        bbox = model(input_image).cpu().numpy()[0]
    
    # Convert normalized coordinates to pixel coordinates
    bbox = bbox * np.array([original_w, original_h, original_w, original_h])
    x_min, y_min, x_max, y_max = bbox
    
    # Visualize result
    fig, ax = plt.subplots(1)
    ax.imshow(original_image)
    
    # Create rectangle patch
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    plt.title('Signature Detection')
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return bbox

if __name__ == '__main__':
    # Example usage
    model_path = 'signature_localizer.pth'
    image_path = 'dataset/val/images/example.jpg'
    predicted_bbox = predict_signature(model_path, image_path, 'output.png')
    print(f"Predicted bounding box: {predicted_bbox}")