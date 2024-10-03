import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    A basic ResNet block contains two convolutional layers with a skip connection.
    3x3 conv -> BatchNorm -> ReLU -> 3x3 conv -> BatchNorm
    '''
    def __init__(self,in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__
        
        self.cn1=nn.Conv2d(in_channels, out_channels, stride=stride, padding=1, kernel_size=3)
        self.bn1=nn.BatchNorm2d(out_channels)
        
        self.cn2=nn.Conv2d(out_channels,out_channels,stride=1, padding=1, kernel_size=3)
        self.bn2=nn.BatchNorm2d(out_channels)
        
        self.relu=nn.ReLU()
        self.downsample=downsample
         
    def forward(self,x):
        identity=x
        
        x=self.cn1(x)
        x=self.bn1(x)
        x=self.relu(x)
        
        x=self.cn2(x)
        x=self.bn2(x)
        
        if self.downsample is not None:
            identity=self.downsample(identity)
        
        x+=identity
        x=self.relu(x)
        
class BottleNeck(nn.Module):
    '''
    A bottleneck ResNet block contains 3 conv layers with a skip connection
    '''
    def __init__(self,in_channels, out_channels, stride, downsample=None):
        super(BottleNeck, self).__init__
        
        self.expansion=4
        
        self.cn1=nn.Conv2d(in_channels, out_channels, stride=1, padding=1, kernel_size=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        
        self.cn2=nn.Conv2d(out_channels, out_channels, stride=stride, padding=1, kernel_size=3)
        self.bn2=nn.BatchNorm2d(out_channels)
        
        
        self.cn3=nn.Conv2d(out_channels,out_channels*self.expansion,stride=1, padding=1, kernel_size=1)
        self.bn3=nn.BatchNorm2d(out_channels)
        
        self.relu=nn.ReLU()
        self.downsample=downsample
         
    def forward(self,x):
        identity=x
        
        x=self.cn1(x)
        x=self.bn1(x)
        x=self.relu(x)
        
        x=self.cn2(x)
        x=self.bn2(x)
        
        if self.downsample is not None:
            identity=self.downsample(identity)
        
        x+=identity
        x=self.relu(x)