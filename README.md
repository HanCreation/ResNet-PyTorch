# ResNet-PyTorch
 Experimenting with ResNet using PyTorch and CIFAR10 Dataset by implementing ResNet from scratch using PyTorch and add the ability to custom the architechture of the resnet blocks. This implementation is based on ResNet50, which uses BottleNeck blocks

## ResNet is Special
[Resnet Paper](https://arxiv.org/abs/1512.03385)

ResNet is special because it introduced residual connections, which allow the network to "skip" layers, making it easier to train very deep networks by avoiding the vanishing gradient problem. 

This architecture enables networks to be much deeper while improving performance, revolutionizing deep learning for tasks like image recognition.

Skip layers (residual connections) work mathematically because they preserve the gradient flow during backpropagation, allowing the model to avoid the vanishing gradient problem. By adding the input 𝑥 directly to the output of a block 

output of a block = F(𝑥)+𝑥

The addition operation ensures that the gradient flows through the network more easily. The skip connection contributes a gradient of 1, allowing the gradient to pass directly through without being diminished, which helps prevent vanishing gradients and makes training deep networks more effective.

## CIFAR10
CIFAR-10 is a dataset for image classification, consisting of 60,000 32x32 color images in 10 different classes (e.g., airplanes, cars, animals), with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images, and is commonly used to benchmark and evaluate the performance of deep learning algorithms in computer vision.

## Data Augmentation
This experiment outlines a new insight for me which is the importance of data augmentation. By randomly modifying each image in training set, the model can capture the feature better and improving the accuracy.

I experimented without the data augmentation ResNet50 can only achieve approximately 75% on the test data.

I added augmentation such as random cropping, padding, horizontal flipping, and random erase to the training set and for more juice I reduced the number of ResNet blocks. But still it makes the model performed better which perfomed best at 87.92% on test data

Coded and Created by Han 2024