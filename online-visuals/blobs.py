import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization

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

# Create interactive visualization (uses auto-calculated ranges)
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Blobs Dataset',
    grid_resolution=config['visualization']['grid_resolution']
)