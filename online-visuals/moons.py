import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate moons dataset
X, y = data.make_moons(
    n_samples=config['training']['n_samples'],
    noise=0.1,
    random_state=config['training']['random_state']
)

# Create interactive visualization with custom ranges for moons
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Moons Dataset',
    x_range=[-1.5, 2.5],  # Adjusted for moons dataset
    y_range=[-1.0, 1.5],  # Adjusted for moons dataset
    grid_resolution=config['visualization']['grid_resolution']
)