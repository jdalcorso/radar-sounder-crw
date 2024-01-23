import torch
import torch.nn as nn
from torchvision.models import resnet18
    
class CNN(nn.Module):
    def __init__(self, prefc = False):
        super(CNN, self).__init__()
        self.prefc = prefc

        # Layer 1: Initial layer with a 5x5 filter
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=1, padding_mode='reflect')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Layer 2: Additional layer with an 5x5 filter
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, padding=1, padding_mode='reflect')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Layer 3: Intermediate layers with 3x3 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.relu3 = nn.ReLU()
        
        # Layer 4: Another layer with 3x3 filter and adjusted stride
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode='reflect')
        self.relu4 = nn.ReLU()
        
        # Layer 5: Final conv with 3x3 filter
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode='reflect')
        self.relu5 = nn.ReLU()

        # Layer 6: Final layer to linearize output
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 128)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

    def forward(self, x):
        # Forward pass through the layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))

        if self.prefc:
            print(x.shape)
            return x

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet(nn.Module):
    def __init__(self, prefc = False):
        super(Resnet, self).__init__()
        self.model = resnet18()
        for name, module in self.model.named_modules():
            print(f"Module name: {name}, Module type: {module}")

    def forward(self, x):
        return x