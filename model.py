"""This file contains functions to create and save the model, load the dataset and to train and evaluate the created model."""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(batch_size: int=64):
    """Loading cifar10 dataset and creating dataloaders for both training and testing of models."""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_set = datasets.CIFAR10(root='~/data',
                                 download=True,
                                 train=True,
                                 transform=transform)
    test_set = datasets.CIFAR10(root='~/data',
                                train=False,
                                download=True,
                                transform=transform)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)
    
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=True)
    
    return train_loader, test_loader

class CNNModel(nn.Module):
    """Create a class for model creation"""
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        # self.conv3 = nn.Conv2d(in_channels=64,
        #                        out_channels=256,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=16*5*5,
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128,
                             out_features=64)
        self.fc3 = nn.Linear(in_features=64,
                             out_features=32)
        self.fc4 = nn.Linear(in_features=32,
                             out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
        
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epochs[{epochs+1}/{epochs}]-Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0., 0.

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = correct/total

    print(f'Test Accuracy: {accuracy*100}%.')

def save_model(model, filename='cifar10_model.pth'):
    torch.save(model.state_dict(), filename)
    print(f'Model is saved as {filename}.')

def main():
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=1e-3)
    train_loader, test_loader = load_dataset()
    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, test_loader)

if __name__=='__main__':
    main()