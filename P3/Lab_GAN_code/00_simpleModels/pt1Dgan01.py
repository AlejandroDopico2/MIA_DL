#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 1000
target_mean = 0
target_std = 1
initial_guess_mean = 5  # Starting mean far to the right
initial_guess_std = 1
learning_rate = 0.01
iterations = 1000
variance_term_weight = 10

# Define a simple neural network for the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(1, 1)  # Simple linear layer
        # Initialize weights
        self.fc.weight.data.fill_(0.0)  # This corresponds to the slope (ignored here since it's a direct mapping)
        self.fc.bias.data.fill_(initial_guess_mean)  # Start from an initial mean far from target

    def forward(self, x):
        return self.fc(x)

# Target Distribution: Normal distribution
def target_distribution():
    return torch.normal(mean=target_mean, std=target_std, size=(num_samples, 1))

# Initialize generator
generator = Generator()
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

# Custom loss function
def custom_loss(gen_samples, target_samples):
    mse_loss = torch.mean((gen_samples - target_samples) ** 2)
    variance_loss = variance_term_weight * (torch.std(gen_samples, unbiased=False) - torch.std(target_samples, unbiased=False)) ** 2
    return mse_loss + variance_loss

# Set up interactive mode
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Fix the x-axis range for the histograms
x_min = -5
x_max = 10
ax1.set_xlim(x_min, x_max)

# Main simulation loop
errors = []
for i in range(iterations):
    optimizer.zero_grad()
    
    # Generate samples
    noise = torch.randn((num_samples, 1))  # Input noise to the generator
    gen_samples = generator(noise)

    # Get target samples
    target_samples = target_distribution()

    # Calculate loss
    loss = custom_loss(gen_samples, target_samples)
    errors.append(loss.item())

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update plots
    ax1.cla()  # Clear the histogram axis
    ax1.hist(target_samples.detach().numpy(), bins=30, alpha=0.5, label='Target Distribution', color='blue')
    ax1.hist(gen_samples.detach().numpy(), bins=30, alpha=0.5, label='Generated Distribution', color='red')
    ax1.legend()
    ax1.set_title('Histogram of Distributions')
    ax1.set_xlim(x_min, x_max)  # Ensure the x-axis does not change

    ax2.cla()  # Clear the error axis
    ax2.plot(errors, label='MSE Loss')
    ax2.legend()
    ax2.set_title('Loss Over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')

    plt.pause(0.1)  # Pause to update the plots

plt.ioff()  # Turn off interactive mode
plt.show()