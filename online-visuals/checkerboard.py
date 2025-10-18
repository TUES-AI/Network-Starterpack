import numpy as np
import yaml
from interactive_mlp import create_interactive_mlp_visualization

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate checkerboard dataset
np.random.seed(config['training']['random_state'])
n_samples = config['training']['n_samples']

# Create checkerboard pattern
X = np.random.uniform(-4, 4, (n_samples, 2))
y = ((np.sin(X[:, 0] * np.pi) > 0) ^ (np.sin(X[:, 1] * np.pi) > 0)).astype(int)

# Create interactive visualization with custom ranges for checkerboard
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Checkerboard Dataset',
    x_range=[-4, 4],  # Wider range for checkerboard
    y_range=[-4, 4],  # Wider range for checkerboard
    grid_resolution=config['visualization']['grid_resolution']
)