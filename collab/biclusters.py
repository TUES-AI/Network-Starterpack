"""
Biclusters dataset visualization for Google Colab.
"""

import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization_with_config

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate biclusters dataset
X, rows, _ = data.make_biclusters(
    shape=(config['training']['n_samples'], 2),
    n_clusters=2,
    noise=0.1,
    random_state=config['training']['random_state']
)

y = rows[0].astype(int)

print("=" * 70)
print("📊 BICLUSTERS DATASET LOADED")
print("=" * 70)
print(f"✅ Generated {len(X)} data points")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"   - Cluster {label}: {count} points")
print("=" * 70)
print()

create_interactive_mlp_visualization_with_config(
    X,
    y,
    title='MLP Decision Boundary on Biclusters Dataset'
)
