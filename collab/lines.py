"""
Linearly separable dataset visualization for Google Colab.
"""

import numpy as np
import yaml
from interactive_mlp import create_interactive_mlp_visualization_with_config

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate linearly separable dataset
np.random.seed(config['training']['random_state'])
n_samples = config['training']['n_samples']

half_samples = n_samples // 2
X1 = np.random.randn(half_samples, 2) + [2, 2]
X2 = np.random.randn(n_samples - half_samples, 2) + [-2, -2]
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(len(X1)), np.ones(len(X2))]).astype(int)

print("=" * 70)
print("📊 LINES DATASET LOADED")
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
    title='MLP Decision Boundary on Lines Dataset',
    config_overrides={
        'x_range': [-5, 5],
        'y_range': [-5, 5]
    }
)
