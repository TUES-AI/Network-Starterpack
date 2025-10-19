"""
Checkerboard dataset visualization for Google Colab.
"""

import numpy as np
import yaml
from interactive_mlp import create_interactive_mlp_visualization_with_config

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate checkerboard dataset
np.random.seed(config['training']['random_state'])
n_samples = config['training']['n_samples']
X = np.random.uniform(-4, 4, (n_samples, 2))
y = ((np.sin(X[:, 0] * np.pi) > 0) ^ (np.sin(X[:, 1] * np.pi) > 0)).astype(int)

print("=" * 70)
print("📊 CHECKERBOARD DATASET LOADED")
print("=" * 70)
print(f"✅ Generated {len(X)} data points")
class_counts = np.bincount(y)
for label, count in enumerate(class_counts):
    print(f"   - Class {label}: {count} points")
print("=" * 70)
print()

create_interactive_mlp_visualization_with_config(
    X,
    y,
    title='MLP Decision Boundary on Checkerboard Dataset',
    config_overrides={
        'x_range': [-4, 4],
        'y_range': [-4, 4]
    }
)
