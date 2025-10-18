import numpy as np
import yaml
from interactive_mlp import create_interactive_mlp_visualization

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate linearly separable dataset
np.random.seed(config['training']['random_state'])
n_samples = config['training']['n_samples']

# Create two linearly separable clusters
X1 = np.random.randn(n_samples//2, 2) + [2, 2]
X2 = np.random.randn(n_samples//2, 2) + [-2, -2]
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

# Create interactive visualization with custom ranges for lines dataset
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Lines Dataset',
    x_range=[-5, 5],  # Wider range for lines dataset
    y_range=[-5, 5],  # Wider range for lines dataset
    grid_resolution=config['visualization']['grid_resolution']
)