"""
Circles dataset visualization for Google Colab
"""

import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp_colab import create_interactive_mlp_visualization_with_config

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate circles dataset
X, y = data.make_circles(
    n_samples=config['training']['n_samples'],
    noise=0.1,
    factor=0.3,
    random_state=config['training']['random_state']
)

print("="*70)
print("📊 CIRCLES DATASET LOADED")
print("="*70)
print(f"✅ Generated {len(X)} data points")
print(f"   - Class 0 (red circles): {(y==0).sum()} points")
print(f"   - Class 1 (blue X's): {(y==1).sum()} points")
print("="*70)
print()

# Create interactive visualization
create_interactive_mlp_visualization_with_config(
    X, y,
    title='MLP Decision Boundary on Circles Dataset'
)
