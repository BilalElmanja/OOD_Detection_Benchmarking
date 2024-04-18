import torch
import torch.nn as nn
import torch.optim as optim
from oodeel.datasets import OODDataset
from oodeel.methods.base import OODBaseDetector
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


import os
import argparse
import time
from contextlib import contextmanager
from PIL import Image

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
import sys
sys.path.append("../")
from oodeel.methods import MLS, Energy, Entropy, DKNN, Gram, Mahalanobis, ODIN, VIM
from methods import  K_Means, PCA_KNN, NMF_KNN, PCA_MAHALANOBIS, NMF_MAHALANOBIS, PCA_unique_class_KNN, PCA_Unique_Class_Mahalanobis, NMF_Unique_Classes_KNN, NMF_Unique_Class_Mahalanobis
from data_preprocessing import get_train_dataset_cifar10, get_test_dataset_cifar10, get_train_dataset_cifar100, get_test_dataset_cifar100, get_test_dataset_places365, get_test_dataset_svhn, get_test_dataset_texture, get_test_dataset_Tiny, get_test_dataset_NINCO, get_test_dataset_OpenImage_O, get_train_dataset_inaturalist, get_test_dataset_SSB_hard
from models import load_pretrained_weights_32
from oodeel.eval.metrics import bench_metrics
from oodeel.datasets import OODDataset
from oodeel.types import List
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.linear(128, latent_dim)
        self.fc4 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3(x)
        log_var = self.fc4(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var


def train_vae(model, train_loader, epochs=100, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(train_loader) as tepoch:
            for features, label in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                # data = data.to(device)  # Ensure data is on the correct device (CPU/GPU)
                optimizer.zero_grad()
                x_reconstructed, mu, log_var = model(features)
                recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, features, reduction='sum')
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_div
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                tepoch.set_postfix(loss=total_loss.item())

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}')


class VAE_OOD_Detector(OODBaseDetector):
    def __init__(self):
        super().__init__()
        self.model = VAE(input_dim=512, latent_dim=64)

    def _fit_to_dataset(self, fit_dataset):
      # we calculate the activations_matrix A_train for the training dataset, in order to calculate the CAVs Matrix
      print("shape : ", fit_dataset.shape)
      training_features = self.feature_extractor.predict(fit_dataset)
      # the activations_matrix A_train
      A_train = training_features[0][0]
      if len(A_train.shape) > 2:
        A_train = A_train[:,:, 0, 0]
        
      print(" shape of A_in is : ", A_train.shape)
      # Create a TensorDataset
      feature_dataset = TensorDataset(A_train)
      # Create a DataLoader
      train_loader = DataLoader(feature_dataset, batch_size=128, shuffle=True)
      # fitting the vae model
      train_vae(self.model, train_loader )
      # eval mode for vae
      self.model.eval()


    def _score_tensor(self, inputs):
        features, logits = self.feature_extractor.predict_tensor(inputs)
        if len(features[0].shape) > 2:
            features[0] = features[0][:,:, 0, 0]
        # self.model.eval()
        with torch.no_grad():
            x_reconstructed, _, _ = self.model(features[0])
            recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, features[0], reduction='none')
            recon_loss = torch.mean(recon_loss, dim=1)
        return recon_loss.cpu().numpy()

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True


ds_train = get_train_dataset_cifar10()
ds_in = get_test_dataset_cifar10()
ds_out = get_test_dataset_svhn()
model = load_pretrained_weights_32()

vae = VAE_OOD_Detector()
vae.fit(model, feature_layers_id=[-2], fit_dataset=ds_train)
