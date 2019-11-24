# Deep Convolutional GANs

## Overview

This application uses a Deep Convolutional GAN (DCGAN) to generate new, realistic-looking images of house numbers.

## Approach

The following steps are taken:
1. Loading and pre-processing the house numbers dataset
2. Defining the discriminator and generator networks
3. Training these adversarial networks
4. Visualization off the loss over time and some sample, generated images

## Dataset

The model is trained on the Street View House Numbers (SVHN) dataset. These are color images of house numbers collected from Google street view.

## How does it work?

The discriminator alternates training on real and fake (generated) images, and the generator aims to trick the discriminator into thinking that its generated images are real.
