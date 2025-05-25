# importing the modules
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import SignatureDataset, get_transform
from model import SignatureLocalizer
import matplotlib.pyplot as plt

# define the train_model function
def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    train_dataset = SignatureDataset(
        root_dir='dataset/train',
        transform=get_transform(train=True)
    )
    
    val_dataset = SignatureDataset(
        root_dir='dataset/val',
        transform=get_transform(train=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SignatureLocalizer(pretrained=True).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, bboxes in train_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        val_loss = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), 'signature_localizer.pth')
    print("Model saved to signature_localizer.pth")

# Function to evaluate_model on validation set
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, bboxes in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

if __name__ == '__main__':
    train_model()