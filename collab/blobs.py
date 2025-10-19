"""
Blobs dataset visualization for Google Colab.
"""

import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization_with_config

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate blobs dataset
X, y = data.make_blobs(
    n_samples=config['training']['n_samples'],
    centers=2,
    cluster_std=1.0,
    random_state=config['training']['random_state']
)

print("=" * 70)
print("📊 BLOBS DATASET LOADED")
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
    title='MLP Decision Boundary on Blobs Dataset'
)
