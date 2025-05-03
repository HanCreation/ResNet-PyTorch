import torch
import torch.nn as nn
from .block import BasicBlock, BottleNeck

class ResNet(nn.Module):
    """
    Implementation of ResNet architecture.
    """
    def __init__(self, block, layers, image_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # For BasicBlock expansion is 1, for BottleNeck it's 4
        expansion = 4 if block == BottleNeck else 1
        self.fc = nn.Linear(512 * expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, out_channels, blocks, stride=1):
        identity_downsample = None
        layers = []

        # For BasicBlock expansion is 1, for BottleNeck it's 4
        expansion = 4 if block == BottleNeck else 1
        
        # Apply downsample if stride != 1 or if in_channels != out_channels * expansion
        if stride != 1 or self.in_channels != out_channels * expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * expansion)
            )

        # Add first block with downsample
        layers.append(block(self.in_channels, out_channels, stride, identity_downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * expansion
        
        # Add remaining blocks
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=None))  # stride=1 for subsequent blocks

        return nn.Sequential(*layers)


def resnet18(num_classes=10, image_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], image_channels=image_channels, num_classes=num_classes)

def resnet34(num_classes=10, image_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)

def resnet50(num_classes=10, image_channels=3):
    return ResNet(BottleNeck, [3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)

def resnet101(num_classes=10, image_channels=3):
    return ResNet(BottleNeck, [3, 4, 23, 3], image_channels=image_channels, num_classes=num_classes)

def resnet152(num_classes=10, image_channels=3):
    return ResNet(BottleNeck, [3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes) 