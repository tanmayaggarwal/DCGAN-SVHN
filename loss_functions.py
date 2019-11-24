# defining the loss functions
# import standard libraries
import torch
import torch.nn as nn

# Define the Discriminator loss
# For the Discriminator, the total loss is the sum of the losses for real and fake images
# Discriminator outputs 1 for real images and 0 for fake images
# We use the BCEWithLogitsLoss function which combines a sigmoid activation function and binary cross entropy loss in one function

# Define the Generator loss
# Generator loss is similar but with flipped labels
# Generator's goal is to get the Discriminator to output 1 for fake images

def real_loss(D_out, train_on_gpu, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1

    # move labels to GPU if available
    if train_on_gpu:
        labels = labels.cuda()

    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)

    return loss

def fake_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss
