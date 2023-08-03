# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:46:19 2023

@author: doguk
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 512 * 20 * 20),
            nn.BatchNorm1d(512 * 20 * 20),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, 20, 20)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 20 * 20, 1), #Matching the generator's first layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def PlotGraph(g_losses, d_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Performance.png")
    plt.show()
        
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = './TrainingImageFolder'
generator_path = 'generator.pth'
discriminator_path = 'discriminator.pth'
dataset = ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

if os.path.exists(generator_path):
    generator.load_state_dict(torch.load(generator_path))
    print('Generator model loaded.')
else:
    print('generator either not existing or couldnt loaded.')
    
if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path))
    print('Discriminator model loaded.')
else:
    print('discriminator either not existing or couldnt loaded.')

# Loss and optimizers
#criterion = nn.BCELoss() replaced
criterion = nn.MSELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001)

# Create a fixed noise vector to visualize the progress of the generator
fixed_noise = torch.randn(64, 128).to(device) # Batch size 64, noise dimension 128
os.makedirs('Training_progress', exist_ok=True)
# Training loop
num_epochs = 50
g_losses = []
d_losses = []
for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0) # Get the current batch size

        real_labels = torch.ones(batch_size, 1).to(device)  # Create real labels based on current batch size
        fake_labels = torch.zeros(batch_size, 1).to(device) # Create fake labels based on current batch size

        discriminator_update_interval = 5
        if epoch % discriminator_update_interval == 0:
            # Train Discriminator
            real_images = real_images.to(device)

            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            noise = torch.randn(batch_size, 128).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 0.01) # Clip gradients
            optimizer_d.step()

    # Train Generator
    noise = torch.randn(batch_size, 128).to(device)
    fake_images = generator(noise)
    # Compute pairwise similarities between generated images
    similarities = torch.pdist(fake_images.view(batch_size, -1), p=2) # Euclidean distance
    # Compute mean similarity
    mean_similarity = torch.mean(similarities)
    # Regular GAN loss
    outputs = discriminator(fake_images)
    g_loss = criterion(outputs, real_labels)
    # Penalize lack of diversity
    lambda_diversity = 0.01 # Hyperparameter to tune
    g_loss += lambda_diversity * mean_similarity

    optimizer_g.zero_grad()
    g_loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), 0.01) # Clip gradients
    optimizer_g.step()
        
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_images = generator(fixed_noise).cpu()
            fake_images = (fake_images + 1) / 2  # Unnormalize
            torchvision.utils.save_image(fake_images, f'Training_progress/epoch_{epoch}.png', nrow=8)  # Save as grid
    
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
 
PlotGraph(g_losses, d_losses)
torch.save(generator.cpu().state_dict(), generator_path)
torch.save(discriminator.cpu().state_dict(), discriminator_path)
print('Generator model saved.')