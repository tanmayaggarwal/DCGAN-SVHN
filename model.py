# defining the models
# import standard libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    # creates a convolution layer, with optional batch normalization

    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)

# define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim

        #32x32 input
        self.conv1 = conv(3, conv_dim, 4, 2, 1, batch_norm=False)
        #16x16 out
        self.conv2 = conv(conv_dim, conv_dim*2, 4, 2, 1, batch_norm=True)
        #8x8 out
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, 2, 1, batch_norm=True)
        #4x4 out

        #final, fully-connected layer
        self.output = nn.Linear(conv_dim*4*4*4, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        # pass x through all layers
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))

        # flatten image
        x = x.view(-1, self.conv_dim*4*4*4)
        x = self.output(x)

        return x

# Generator network architecture

# first layer is a fully connected layer which is reshaped into a deep and narrow layer
# Next is a series of transpose convolutional layers (halve the depth and double the width and height of the previous layer)
# Apply batch normalization and ReLU to all but the last of the hidden layers
# tanh is applied to the last layer and the output will be of size 32x32

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    # creates a transposed-convolutional layer, with optional batch normalization
    layers = []
    conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)

# define the Generator network
class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        # first, fully-connected layer
        self.fc1 = nn.Linear(z_size, conv_dim*4*4*4)
        # transpose conv layers
        self.tconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.tconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.tconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        # fully-connected + reshape
        x = self.fc1(x)
        x = x.view(-1, self.conv_dim*4, 4, 4) # (batch_size, depth, 4, 4)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = self.tconv3(x)
        x = F.tanh(x)

        return x
