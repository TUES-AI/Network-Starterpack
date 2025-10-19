"""
Moons dataset visualization for Google Colab.
"""

import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization_with_config

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate moons dataset
X, y = data.make_moons(
    n_samples=config['training']['n_samples'],
    noise=0.1,
    random_state=config['training']['random_state']
)

print("=" * 70)
print("📊 MOONS DATASET LOADED")
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
    title='MLP Decision Boundary on Moons Dataset',
    config_overrides={
        'x_range': [-1.5, 2.5],
        'y_range': [-1.0, 1.5]
    }
)
