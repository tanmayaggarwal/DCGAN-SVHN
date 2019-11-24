# loading the data
# import standard libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

def load_data():
    # tensor transform
    transform = transforms.ToTensor()

    # SVHN training datasets
    svhn_train = datasets.SVHN(root='data/', split='train', download=True, transform=transform)

    batch_size = 128
    num_workers = 0

    # build DataLoaders for SVHN dataset
    train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, batch_size, num_workers

def visualize_data(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # plot the images in the batch, along with corresponding labels
    fig = plt.figure(figsize=(25, 4))
    plot_size = 20
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[idx], (1,2,0)))
        # print out the correct label for each image
        ax.set_title(str(labels[idx].item()))

    return images, labels
