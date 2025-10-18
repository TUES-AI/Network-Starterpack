import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate gaussian quantiles dataset
X, y = data.make_gaussian_quantiles(
    n_samples=config['training']['n_samples'],
    n_features=2,
    n_classes=2,
    random_state=config['training']['random_state']
)

# Create interactive visualization
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Gaussian Quantiles Dataset',
    grid_resolution=config['visualization']['grid_resolution']
)