import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate classification dataset
X, y = data.make_classification(
    n_samples=config['training']['n_samples'],
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=config['training']['random_state']
)

# Create interactive visualization
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Classification Dataset',
    grid_resolution=config['visualization']['grid_resolution']
)