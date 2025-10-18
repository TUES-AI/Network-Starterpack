import numpy as np
import sklearn.datasets as data
import yaml
from interactive_mlp_clickable import create_interactive_mlp_visualization_clickable_with_config

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

# Create interactive visualization with click-to-add functionality
create_interactive_mlp_visualization_clickable_with_config(
    X, y,
    title='Interactive MLP - Click to Add Points'
)

print("\n=== INTERACTIVE MODE ACTIVATED ===")
print("Left click: Add Class 0 point (red circles)")
print("Right click: Add Class 1 point (blue crosses)")
print("Model retrains automatically when points are added")
print("Use 'CLEAR ADDED POINTS' to reset user additions")