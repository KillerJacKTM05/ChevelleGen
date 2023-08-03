# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:01:30 2023

@author: doguk
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
    
    
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval() # Set to evaluation mode
num_batches = int(input("Enter the number of batches you want to generate: "))
images_per_batch = int(input("Enter the number of images per batch: "))
for batch_num in range(num_batches):
    noise = torch.randn(images_per_batch, 100)
    fake_images = generator(noise).detach()
    
    for image_num in range(images_per_batch):
        image = fake_images[image_num].permute(1, 2, 0).numpy() * 0.5 + 0.5
        plt.imsave(f"generated_image_batch{batch_num}_image{image_num}.png", image)

print('Images generated and saved.')

