## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #
        self.conv1 = nn.Conv2d(1, 64, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # conv layers
        self.conv2 = nn.Conv2d(64,128,4,stride = 2) #decreasing kernel size to reduce parameter
        self.conv3 = nn.Conv2d(128,256,4,stride = 2)
        self.conv4 = nn.Conv2d(256,512,3)
        
        
        #adding max pool layer which we are going to use after each conv operation
        self.pool = nn.MaxPool2d(2,2)
        
        #adding dropout layer
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.4)
        self.dropout6 = nn.Dropout(p=0.4)
        
        #adding batchnorm 
        self.bn1 = nn.BatchNorm2d(num_features = 64, eps = 1e-05)
        self.bn2 = nn.BatchNorm2d(num_features = 128, eps = 1e-05)
        self.bn3 = nn.BatchNorm2d(num_features = 256, eps = 1e-05)
        self.bn4 = nn.BatchNorm2d(num_features = 512, eps = 1e-05)
        self.bn5 = nn.BatchNorm1d(num_features=1600, eps=1e-05)
        self.bn6 = nn.BatchNorm1d(num_features=800, eps=1e-05)
        
        #adding fully connected layer
        self.fc1 = nn.Linear(2048,1600)
        self.fc2 = nn.Linear(1600,800)
        self.fc3 = nn.Linear(800,136)
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.elu(self.conv1(x)))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.pool(F.elu(self.conv2(x)))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.pool(F.elu(self.conv3(x)))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.pool(F.elu(self.conv4(x)))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        #converting it into row array
        x = x.view(x.size(0),-1)
        
        x = F.elu(self.fc1(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        
        x = F.elu(self.fc2(x))
        x = self.bn6(x)
        x = self.dropout6(x)
        
        x = F.elu(self.fc3(x))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

