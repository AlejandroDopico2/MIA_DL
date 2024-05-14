#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 1000
target_mean = 0
target_std = 1
initial_guess_mean = 5
initial_guess_std = 1
learning_rate = 0.05   #  0.01
iterations = 100
variance_term_weight = 10 


# Target Distribution: Normal distribution
def target_distribution():
    return np.random.normal(target_mean, target_std, num_samples)

# Generator: Initially guesses a different distribution
def generate_samples(mean, std):
    return np.random.normal(mean, std, num_samples)

# Discriminator: Calculates the Mean Squared Error to the target
def discriminator(gen_samples, target_samples):
    return np.mean((gen_samples - target_samples) ** 2)

# Initialize parameters
mean = initial_guess_mean
std = initial_guess_std

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
    target_samples = target_distribution()
    gen_samples = generate_samples(mean, std)

    # Calculate error
    error = discriminator(gen_samples, target_samples)
    errors.append(error)


    mean_gradient = 2 * np.mean(gen_samples - target_samples)
    std_gradient = 2 * np.mean((gen_samples - target_samples) * (gen_samples - mean) / std) + variance_term_weight * (np.std(gen_samples) - np.std(target_samples))

    mean -= learning_rate * mean_gradient
    std -= learning_rate * std_gradient

    # Update plots
    ax1.cla()  # Clear the histogram axis
    ax1.hist(target_samples, bins=30, alpha=0.5, label='Target Distribution', color='blue')
    ax1.hist(gen_samples, bins=30, alpha=0.5, label='Generated Distribution', color='red')
    ax1.legend()
    ax1.set_title('Histogram of Distributions')
    ax1.set_xlim(x_min, x_max)  # Ensure the x-axis does not change

    # Update statistics textbox with LaTeX
    stats_text = f'$\mu = {mean:.2f}$\n$\sigma = {std:.2f}$'
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    ax2.cla()  # Clear the error axis
    ax2.plot(errors, label='MSE Error')
    ax2.legend()
    ax2.set_title('Error Over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE Error')

    plt.pause(0.1)  # Pause to update the plots

    # Save the frame
    #fig.savefig(f'imgs/frame_{i:04d}.png')  # Saves each frame in a 'frames' directory, numbered with zero padding



plt.ioff()  # Turn off interactive mode
plt.show()