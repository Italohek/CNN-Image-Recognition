import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Preparing the transforms for the dataset (augmentation for training set) 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the CNN architecture
class NeuralNet(nn.Module):

    # Define the layers of the CNN
    def __init__(self):
        super().__init__()

        # Input image size is (3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # (3, 32, 32) -> (16, 32, 32)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (16, 16, 16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # (16, 16, 16) -> (32, 16, 16)
        self.bn2 = nn.BatchNorm2d(32)
        # -> (32, 8, 8)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # (32, 8, 8) -> (64, 8, 8)
        self.bn3 = nn.BatchNorm2d(64)
        # -> (64, 4, 4)

        # Fully connected layers after flattening 
        self.fc1 = nn.Linear(in_features=(64 * 4 * 4), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=5)

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # The bn layer normalizes the output of the conv layer
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # And the relu layer introduces non-linearity
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # The pool layer reduces the spatial dimensions
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) # No activation here since our loss function applies Softmax
        return x

if __name__ == '__main__':
    train_dataset = torchvision.datasets.ImageFolder(root='./Rice_Image_Dataset/', transform=transform) 
    val_test_dataset = torchvision.datasets.ImageFolder(root='./Rice_Image_Dataset/', transform=test_transform)

    # Splitting the dataset into training, validation, and test sets
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    # Create a list of all indices and shuffle them for randomness 
    indices = list(range(len(train_dataset)))

    # Set a random seed for reproducibility and use it in the generator to ensure consistent splits
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(indices, [train_size, val_size, test_size], generator=generator)

    # Create data loaders for each subset
    train_data_final = torch.utils.data.Subset(train_dataset, train_indices)
    val_data_final = torch.utils.data.Subset(val_test_dataset, val_indices)
    test_data_final = torch.utils.data.Subset(val_test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_data_final, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data_final, batch_size=32, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data_final, batch_size=32, shuffle=False, num_workers=2)

    # Dataset statistics
    print(f"Found {len(train_dataset)} images belonging to {len(train_dataset.classes)} classes.")
    print("Classes:", train_dataset.classes)
    print(f"Training set size: {len(train_data_final)}")
    print(f"Validation set size: {len(val_data_final)}")
    print(f"Test set size: {len(test_data_final)}")

    # Initialize the network, loss function, and optimizer
    net = NeuralNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr= 0.001, momentum=0.9)

    # Setting up lists to track losses for plotting and starting the training process
    train_losses = []
    val_losses = []

    for epoch in range(40):
        net.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device) # Move data to the appropriate device
            
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = net(inputs) # Forward pass 
            loss = loss_function(outputs, labels) # Compute the loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            running_loss += loss.item() # Accumulate loss for the epoch
        
        epoch_train_loss = running_loss / len(train_loader) # Average loss for the epoch
        train_losses.append(epoch_train_loss) # Store training loss

        net.eval() 
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        # Validation loop
        with torch.no_grad():
            for data in val_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = loss_function(outputs, labels)
                val_running_loss += loss.item() # Accumulate validation loss
                _, predicted = torch.max(outputs.data, 1) 
                total_val += labels.size(0) # Count total labels
                correct_val += (predicted == labels).sum().item() # Count correct predictions
        
        epoch_val_loss = val_running_loss / len(val_loader) # Average validation loss
        val_losses.append(epoch_val_loss) # Store validation loss
        val_accuracy = 100 * correct_val / total_val # Calculate validation accuracy

        print(f'Epoch {epoch + 1}/{40} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Accuracy: {val_accuracy:.2f}%')

    print('Finished Training')
    torch.save(net.state_dict(), 'trained_net.pth')

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    print("Loss plot saved as loss_plot.png")

    # Testing the network
    net = NeuralNet().to(device)
    net.load_state_dict(torch.load('trained_net.pth'))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print test accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {test_size} test images: {accuracy:.2f}%')