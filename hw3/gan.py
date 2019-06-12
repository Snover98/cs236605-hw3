from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        self.feature_extractor = nn.Sequential(
            # 64 -> 60
            nn.Conv2d(in_channels=in_size[0], out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # 60 -> 30
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, dilation=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),

            # 30 -> 22
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, dilation=2, padding=0),
            nn.LeakyReLU(0.2),
            # 22 -> 11
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, dilation=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            # 11 -> 8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, dilation=3, padding=0),
            nn.LeakyReLU(0.2),
            # 8- > 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, dilation=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            # 4 -> 1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, dilation=1, padding=0),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(

            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x).view(x.shape[0], -1)
        y = self.classifier(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======

        self.conv = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose2d(out_channels=64, in_channels=z_dim, kernel_size=featuremap_size),
            nn.ReLU(),

            nn.BatchNorm2d(64),
            # 4 -> 8
            nn.ConvTranspose2d(out_channels=64, in_channels=64, kernel_size=2, stride=2, dilation=1, padding=0),
            nn.ReLU(),
            # 8 -> 11
            nn.ConvTranspose2d(out_channels=32, in_channels=64, kernel_size=2, stride=1, dilation=3, padding=0),
            nn.ReLU(),

            nn.BatchNorm2d(32),
            # 11 -> 22
            nn.ConvTranspose2d(out_channels=32, in_channels=32, kernel_size=4, stride=2, dilation=1, padding=1),
            nn.ReLU(),
            # 22 -> 30
            nn.ConvTranspose2d(out_channels=16, in_channels=32, kernel_size=5, stride=1, dilation=2, padding=0),
            nn.ReLU(),

            nn.BatchNorm2d(16),
            # 30 -> 60
            nn.ConvTranspose2d(out_channels=16, in_channels=16, kernel_size=4, stride=2, dilation=1, padding=1),
            nn.ReLU(),
            # 60 -> 64
            nn.ConvTranspose2d(out_channels=out_channels, in_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Tanh()

        )

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        noise = torch.randn(n, self.z_dim, requires_grad=with_grad, device=device)
        if not with_grad:
            with torch.no_grad():
                samples = self.forward(noise)
        else:
            samples = self.forward(noise)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.conv(z.view(*z.shape, 1, 1))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device

    num_samples = y_data.shape[0]
    noise_range = (-label_noise / 2, label_noise / 2)

    data_labels = torch.zeros(num_samples) + data_label + torch.FloatTensor(num_samples).uniform_(*noise_range)
    gen_labels = torch.ones(num_samples) - data_label + torch.FloatTensor(num_samples).uniform_(*noise_range)

    data_labels, gen_labels = data_labels.to(device), gen_labels.to(device)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    loss_data = torch.mean(loss_fn(y_data, data_labels))
    loss_generated = torch.mean(loss_fn(y_generated, gen_labels))

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device

    target_labels = torch.zeros_like(y_generated, device=device) + data_label

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    loss = loss_fn(y_generated, target_labels)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    gen_data = gen_model.sample(x_data.shape[0])

    data_scores = torch.squeeze(dsc_model(x_data))
    gen_scores = torch.squeeze(dsc_model(gen_data))

    dsc_loss = dsc_loss_fn(data_scores, gen_scores)

    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    gen_data = gen_model.sample(x_data.shape[0], with_grad=True)
    gen_scores = dsc_model(gen_data)

    gen_loss = gen_loss_fn(gen_scores)

    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()
