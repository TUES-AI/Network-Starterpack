# online-visuals/circles_colab.py
import numpy as np
import yaml
from sklearn.datasets import make_circles
from interactive_mlp_colab import create_interactive_mlp_visualization_with_config

with open('Config.yml', 'r') as f:
    config = yaml.safe_load(f)

X, y = make_circles(
    n_samples=config['training']['n_samples'],
    noise=0.1,
    factor=0.3,
    random_state=config['training']['random_state'],
)

create_interactive_mlp_visualization_with_config(
    X, y, title='MLP decision boundary on Circles Dataset (Colab)'
)

