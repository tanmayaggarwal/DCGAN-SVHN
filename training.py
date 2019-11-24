# defining the training function
# import standard libraries
import numpy as np
import pickle as pkl
import torch
from process_data import scale
from loss_functions import real_loss, fake_loss

# Discriminator Training
# 1. Compute the discriminator loss on real, training images
# 2. Generate fake images
# 3. Compute the discriminator loss on fake, generated images
# 4. Add up real and fake loss
# 5. Perform backpropagation and an optimization step to update the discriminator's weights

# Generator training
# 1. Generate fake images
# 2. Compute the discriminator loss on fake images, using flipped labels
# 3. Perform backpropagation and an optimization step to update the generator's weights

def training(num_epochs, z_size, train_loader, d_optimizer, g_optimizer, D, G, train_on_gpu):
    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 300

    # get some fixed data for sampling
    # these are images that are held constant throughout training, and allow us to inspect the model's performance

    sample_size = 16
    fixed_z = np.random.uniform(-1,1,size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)

            # rescale input image
            real_images = scale(real_images)

            #############################
            ## train the discriminator ##
            #############################

            d_optimizer.zero_grad()

            # 1. Train with real images
            # compute the discriminator loss on real images
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)
            d_real_loss = real_loss(D_real, train_on_gpu)

            # 2. Train with fake images
            # generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # 3. Compute the discriminator losses on fake images
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake, train_on_gpu)

            # 4. Add up losses and 5. perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            #############################
            ## train the generator ##
            #############################

            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels
            # generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # 2. Compute the discriminator loss on fake images using flipped labels
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, train_on_gpu) # use real loss to flip labels

            # 3. perform backprop
            g_loss.backward()
            g_optimizer.step()

            # print loss statistics
            if batch_i % print_every == 0:
                # append discriminator loss and Generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}]  |  d_loss: {:6.4f}  |  g_loss = {:6.4f}'.format(epoch+1, num_epochs, d_loss.item(), g_loss.item()))

        # after each epoch
        # generate and save sample, fake images
        G.eval() # for generating samples
        if train_on_gpu:
            fixed_z = fixed_z.cuda()

        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()

    # save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return D, G, losses, samples
