# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:59:07 2025

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

# Data for 3 models (A, B, C)
models = ['llama3:8b', 'mxbai-embed-large', 'nomic-embed-text']

# Response times (3 runs per model)
times = {
    'llama3:8b': [50,46,52],
    'mxbai-embed-large': [100,99,92],
    'nomic-embed-text': [90,83,82]
}

# Accuracies (3 runs per model)
accuracies = {
    'llama3:8b': [50, 41, 58],
    'mxbai-embed-large': [66, 75, 75],
    'nomic-embed-text': [100,100,91]
}

# Compute means and standard deviations
time_means = [np.mean(times[model]) for model in models]
time_stds = [np.std(times[model]) for model in models]

accuracy_means = [np.mean(accuracies[model]) for model in models]
accuracy_stds = [np.std(accuracies[model]) for model in models]

# Create figure with two subplots
plt.figure(figsize=(12, 6))

# First subplot - Response Time
plt.subplot(1, 2, 1)
bars1 = plt.bar(models, time_means, yerr=time_stds, 
                color='skyblue', capsize=5, edgecolor='black')
plt.xlabel('Models')
plt.ylabel('Response Time (s)')
plt.title('Model Response Times (±1 SD)')

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom')

# Second subplot - Accuracy
plt.subplot(1, 2, 2)
bars2 = plt.bar(models, np.array(accuracy_means), yerr=np.array(accuracy_stds),
                color='lightgreen', capsize=5, edgecolor='black')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracies (±1 SD)')

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()