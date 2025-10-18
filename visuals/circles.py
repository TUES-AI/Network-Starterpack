import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization_with_config

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

# Create interactive visualization
create_interactive_mlp_visualization_with_config(
    X, y,
    title='MLP decision boundary on Circles Dataset'
)