# main application file
# import standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import torch.optim as optim

# %matplotlib inline

# getting the data
from load_data import load_data
train_loader, batch_size, num_workers = load_data()

# visualize the data
from load_data import visualize_data
images, labels = visualize_data(train_loader)

# pre-processing the data
from process_data import process_data
scaled_img = process_data(images)

# define the model
from model import Discriminator, Generator

# define hyperparamaters
conv_dim = 32
z_size = 100

# define discriminator and generator
D = Discriminator(conv_dim)
G = Generator(z_size=z_size, conv_dim=conv_dim)

print(D)
print()
print(G)

# training on GPU if available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    G.cuda()
    D.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')

# defining the loss functions
from loss_functions import real_loss, fake_loss

# define the training hyperparameters
lr = 0.001
beta1 = 0.5
beta2 = 0.999
num_epochs = 30

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

# training the models
from training import training
D, G, losses, samples = training(num_epochs, z_size, train_loader, d_optimizer, g_optimizer, D, G, train_on_gpu)

# plotting the training losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label = 'Discriminator', alpha=0.5)
plt.plot(losses.T[1], label = 'Generator', alpha=0.5)
plt.title("Training losses")
plt.legend()

# saving the models
torch.save({'Discriminator_state_dict': D.state_dict(), 'Generator_state_dict': G.state_dict(), 'D_optimizer_state_dict': d_optimizer.state_dict(), 'G_optimizer_state_dict': g_optimizer.state_dict()}, "saved_model.pth")

# generating samples from training
from view_samples import view_samples
_ = view_samples(-1, samples)
