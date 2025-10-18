import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp import create_interactive_mlp_visualization

# Load shared configuration
with open('Config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate biclusters dataset
X, rows, columns = data.make_biclusters(
    shape=(config['training']['n_samples'], 2),
    n_clusters=2,
    noise=0.1,
    random_state=config['training']['random_state']
)

# Create labels based on row clusters
y = rows[0]  # Use the first row cluster as labels

# Create interactive visualization
create_interactive_mlp_visualization(
    X, y,
    title='MLP decision boundary on Biclusters Dataset',
    grid_resolution=config['visualization']['grid_resolution']
)