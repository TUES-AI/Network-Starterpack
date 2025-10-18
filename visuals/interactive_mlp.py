"""
Centralized interactive MLP visualization module.
Provides reusable functions for creating interactive MLP visualizations with training controls.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from sklearn.neural_network import MLPClassifier
import yaml


def create_interactive_mlp_visualization(X, y, title, x_range=None, y_range=None, grid_resolution=400):
    """
    Create an interactive MLP visualization with training controls.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Input features
    y : array-like, shape (n_samples,)
        Target labels
    title : str
        Plot title
    x_range : tuple, optional
        X-axis range for visualization grid (min, max)
    y_range : tuple, optional
        Y-axis range for visualization grid (min, max)
    grid_resolution : int
        Resolution for the visualization grid

    Returns:
    --------
    None
    """

    # Load shared configuration
    with open('Config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Create figure with interactive controls
    fig = plt.figure(figsize=(8, 7))
    ax = plt.axes([0.1, 0.3, 0.8, 0.65])  # More space for main plot

    # Create input fields - moved to the right for better visibility
    ax_epoch = plt.axes([0.15, 0.15, 0.2, 0.04])
    ax_lr = plt.axes([0.6, 0.15, 0.2, 0.04])  # Moved further right for better centering
    ax_network = plt.axes([0.15, 0.08, 0.5, 0.04])
    ax_button = plt.axes([0.75, 0.08, 0.2, 0.04])

    epoch_field = widgets.TextBox(ax_epoch, 'Epochs:', initial=str(config['network']['max_iter']))
    lr_field = widgets.TextBox(ax_lr, 'Learning Rate:', initial='0.001')
    network_field = widgets.TextBox(ax_network, 'Network Size:', initial=' '.join(map(str, config['network']['hidden_layer_sizes'])))
    train_button = widgets.Button(ax_button, 'TRAIN')

    # Create grid for visualization
    if x_range is None:
        x_range = [X[:, 0].min() - 1, X[:, 0].max() + 1]
    if y_range is None:
        y_range = [X[:, 1].min() - 1, X[:, 1].max() + 1]

    xx = np.linspace(x_range[0], x_range[1], grid_resolution)
    yy = np.linspace(y_range[0], y_range[1], grid_resolution)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    # Global variables for the classifier and plot elements
    clf = None
    contourf = None
    contour = None
    scatter0 = None
    scatter1 = None

    def train_and_plot(event=None):
        nonlocal clf, contourf, contour, scatter0, scatter1

        # Get values from input fields
        try:
            epochs = int(epoch_field.text)
            learning_rate = float(lr_field.text)
            network_sizes = [int(x) for x in network_field.text.split()]
        except ValueError:
            print("Invalid input values. Using defaults.")
            epochs = config['network']['max_iter']
            learning_rate = 0.001
            network_sizes = config['network']['hidden_layer_sizes']

        # Create classifier with custom parameters
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(network_sizes),
            activation=config['network']['activation'],
            solver=config['network']['solver'],
            batch_size=config['network']['batch_size'],
            max_iter=epochs,
            learning_rate_init=learning_rate,
            random_state=config['network']['random_state']
        )

        # Train the model
        clf.fit(X, y)

        # Predict on grid
        Z = clf.predict_proba(grid)[:,1].reshape(XX.shape)

        # Clear previous plot elements
        if contourf:
            contourf.remove()
        if contour:
            contour.remove()
        if scatter0:
            scatter0.remove()
        if scatter1:
            scatter1.remove()

        # Create new plot elements
        contourf = ax.contourf(XX, YY, Z, levels=20, cmap='RdYlBu', alpha=0.4)
        contour = ax.contour(XX, YY, Z, levels=[0.5], colors='purple', linewidths=2)
        scatter0 = ax.scatter(X[y==0,0], X[y==0,1], marker='x', label='class 0')
        scatter1 = ax.scatter(X[y==1,0], X[y==1,1], marker='o', label='class 1', edgecolors='k')

        ax.legend()
        plt.draw()

    # Connect the button to the training function
    train_button.on_clicked(train_and_plot)

    # Initial training and plot
    train_and_plot()

    plt.show()


def create_interactive_mlp_visualization_with_config(X, y, title, config_overrides=None):
    """
    Create an interactive MLP visualization using configuration from Config.yml.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Input features
    y : array-like, shape (n_samples,)
        Target labels
    title : str
        Plot title
    config_overrides : dict, optional
        Override specific configuration values

    Returns:
    --------
    None
    """

    # Load shared configuration
    with open('Config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if key in config['visualization']:
                config['visualization'][key] = value

    x_range = config['visualization']['x_range']
    y_range = config['visualization']['y_range']
    grid_resolution = config['visualization']['grid_resolution']

    create_interactive_mlp_visualization(
        X, y, title, x_range, y_range, grid_resolution
    )