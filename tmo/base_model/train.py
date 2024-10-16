# by yhpark 2024-10-15
# TensorRT Model Optimization PTQ example
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from utils import *

# Function to load training and validation data
def load_data(batch_size, dataset_base_path, transform):
    train_dataset = datasets.ImageFolder(root=f'{dataset_base_path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{dataset_base_path}/val', transform=transform)

    class_to_idx = train_dataset.class_to_idx
    file_path = f'{dataset_base_path}/class_to_idx.json'
    write_json(class_to_idx, file_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Function to load the model
def load_model(num_classes=100):
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    return model

# Function to save the plot as a file
def save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path):
    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # Save the graph as a file
    plot_path = os.path.join(train_progress_dir_path, 'loss_accuracy.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory


best_val_acc = 0.0  # Criterion for saving the best model
# Function to save the best model
def save_checkpoint(model, epoch, val_acc, checkpoint_dir_path, suffix = ''):
    global best_val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = os.path.join(checkpoint_dir_path, f'best_model{suffix}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.2f}%")

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Model prediction and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    return train_loss, train_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct_val / total_val
    return val_loss, val_acc

# Main training loop
def main():
    set_random_seed()
   
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    print(f"current file path: {current_file_path}")
    print(f"current directory: {current_directory}")

    # Hyperparameter settings
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    best_val_acc = 0.0  # Criterion for saving the best model
    checkpoint_dir_path = f'{current_directory}/checkpoint'  # Directory to save graphs and models
    train_progress_dir_path = f'{current_directory}/train_progress'  # Directory to save graphs and models
    dataset_base_path = f'{current_directory}/../datasets/imagenet100'

    # Create a folder to store results
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    if not os.path.exists(train_progress_dir_path):
        os.makedirs(train_progress_dir_path)

    # Define data preprocessing (for ImageNet100)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Set up model, data, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data(batch_size, dataset_base_path, transform)
    model = load_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize lists to store training progress
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print training and validation info
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Save the best model
        save_checkpoint(model, epoch, val_acc, checkpoint_dir_path)

        # Save the graph as a file
        save_plot(train_losses, val_losses, train_accuracies, val_accuracies, epoch, train_progress_dir_path)

    print("Training complete.")

# Execute the main function
if __name__ == "__main__":
    main()
