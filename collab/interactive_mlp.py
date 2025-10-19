"""
Enhanced interactive MLP visualization module for Google Colab.
Allows users to add Class 0 and Class 1 points to test model generalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import yaml
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import HBox, VBox


def create_interactive_mlp_visualization(X, y, title, x_range=None, y_range=None, grid_resolution=400):
    """
    Create an interactive MLP visualization with click-to-add data points for Google Colab.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Original input features
    y : array-like, shape (n_samples,)
        Original target labels
    title : str
        Plot title
    x_range : tuple, optional
        X-axis range for visualization grid (min, max)
    y_range : tuple, optional
        Y-axis range for visualization grid (min, max)
    grid_resolution : int
        Resolution for the visualization grid
    """

    # Load shared configuration
    with open('Config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Create grid for visualization
    if x_range is None:
        x_range = [X[:, 0].min() - 1, X[:, 0].max() + 1]
    if y_range is None:
        y_range = [X[:, 1].min() - 1, X[:, 1].max() + 1]

    xx = np.linspace(x_range[0], x_range[1], grid_resolution)
    yy = np.linspace(y_range[0], y_range[1], grid_resolution)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    # Store original data
    X_original = X.copy()
    y_original = y.copy()

    # Store user-added points
    state = {
        'X_user_class0': [],
        'X_user_class1': [],
        'clf': None
    }

    # Create ipywidgets controls
    epoch_field = widgets.IntText(
        value=config['network']['max_iter'],
        description='Epochs:',
        style={'description_width': '100px'}
    )
    
    lr_field = widgets.FloatText(
        value=0.001,
        description='Learning Rate:',
        style={'description_width': '100px'},
        step=0.0001,
        format='.4f'
    )
    
    network_field = widgets.Text(
        value=' '.join(map(str, config['network']['hidden_layer_sizes'])),
        description='Network Size:',
        style={'description_width': '100px'}
    )
    
    train_button = widgets.Button(
        description='TRAIN',
        button_style='success',
        tooltip='Train the network with current settings'
    )
    
    clear_button = widgets.Button(
        description='CLEAR ADDED POINTS',
        button_style='warning',
        tooltip='Remove all user-added points'
    )
    
    # Input fields for manual point addition
    add_x_field = widgets.FloatText(
        value=0.0,
        description='X:',
        style={'description_width': '30px'},
        step=0.1
    )
    
    add_y_field = widgets.FloatText(
        value=0.0,
        description='Y:',
        style={'description_width': '30px'},
        step=0.1
    )
    
    add_class0_button = widgets.Button(
        description='Add Class 0',
        button_style='danger',
        tooltip='Add a Class 0 point at (X, Y)'
    )
    
    add_class1_button = widgets.Button(
        description='Add Class 1',
        button_style='info',
        tooltip='Add a Class 1 point at (X, Y)'
    )

    output = widgets.Output()
    plot_output = widgets.Output()

    def train_and_plot(event=None):
        # Get values from input fields
        try:
            epochs = int(epoch_field.value)
            learning_rate = float(lr_field.value)
            network_sizes = [int(x) for x in network_field.value.split()]
        except ValueError:
            with output:
                print("⚠️ Invalid input values. Using defaults.")
            epochs = config['network']['max_iter']
            learning_rate = 0.001
            network_sizes = config['network']['hidden_layer_sizes']

        # Combine original and user-added data
        X_combined = X_original.copy()
        y_combined = y_original.copy()

        if state['X_user_class0']:
            X_combined = np.vstack([X_combined, np.array(state['X_user_class0'])])
            y_combined = np.hstack([y_combined, np.zeros(len(state['X_user_class0']))])

        if state['X_user_class1']:
            X_combined = np.vstack([X_combined, np.array(state['X_user_class1'])])
            y_combined = np.hstack([y_combined, np.ones(len(state['X_user_class1']))])

        with output:
            output.clear_output(wait=True)
            print(f"🔄 Training with {len(X_combined)} points...")
            print(f"   Added points: {len(state['X_user_class0'])} Class 0, {len(state['X_user_class1'])} Class 1")

        # Create classifier with custom parameters
        state['clf'] = MLPClassifier(
            hidden_layer_sizes=tuple(network_sizes),
            activation=config['network']['activation'],
            solver=config['network']['solver'],
            batch_size=config['network']['batch_size'],
            max_iter=epochs,
            learning_rate_init=learning_rate,
            random_state=config['network']['random_state']
        )

        # Train the model
        state['clf'].fit(X_combined, y_combined)
        
        with output:
            print(f"✅ Training complete! Final loss: {state['clf'].loss_:.4f}")

        # Predict on grid
        Z = state['clf'].predict_proba(grid)[:,1].reshape(XX.shape)

        # Create plot
        with plot_output:
            plot_output.clear_output(wait=True)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot decision boundary
            contourf = ax.contourf(XX, YY, Z, levels=20, cmap='RdYlBu', alpha=0.5)
            contour = ax.contour(XX, YY, Z, levels=[0.5], colors='purple', linewidths=2)
            cbar = plt.colorbar(contourf, ax=ax, label='Class 1 Probability')

            # Plot original data points
            ax.scatter(X_original[y_original==0,0], X_original[y_original==0,1],
                      marker='o', label='Class 0 (original)', color='red', alpha=0.8, s=60, edgecolors='darkred')
            ax.scatter(X_original[y_original==1,0], X_original[y_original==1,1],
                      marker='x', label='Class 1 (original)', color='blue', alpha=0.8, s=60, linewidths=2)

            # Plot user-added points
            if state['X_user_class0']:
                ax.scatter(np.array(state['X_user_class0'])[:,0], np.array(state['X_user_class0'])[:,1],
                          marker='o', label='Class 0 (added)', color='red', s=200, 
                          edgecolors='black', linewidths=3, zorder=5)
            if state['X_user_class1']:
                ax.scatter(np.array(state['X_user_class1'])[:,0], np.array(state['X_user_class1'])[:,1],
                          marker='x', label='Class 1 (added)', color='blue', s=200, 
                          linewidths=4, zorder=5)

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_xlabel('Feature 1', fontsize=12)
            ax.set_ylabel('Feature 2', fontsize=12)
            ax.set_title(f'{title}\nAdded: {len(state["X_user_class0"])} Class 0, {len(state["X_user_class1"])} Class 1 points', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def add_class0_point(event):
        x_val = add_x_field.value
        y_val = add_y_field.value
        state['X_user_class0'].append([x_val, y_val])
        with output:
            output.clear_output(wait=True)
            print(f'✅ Added Class 0 point at ({x_val:.2f}, {y_val:.2f})')
            print(f'   Total added: {len(state["X_user_class0"])} Class 0, {len(state["X_user_class1"])} Class 1')
            print('   Click TRAIN to see the updated model')

    def add_class1_point(event):
        x_val = add_x_field.value
        y_val = add_y_field.value
        state['X_user_class1'].append([x_val, y_val])
        with output:
            output.clear_output(wait=True)
            print(f'✅ Added Class 1 point at ({x_val:.2f}, {y_val:.2f})')
            print(f'   Total added: {len(state["X_user_class0"])} Class 0, {len(state["X_user_class1"])} Class 1')
            print('   Click TRAIN to see the updated model')

    def clear_user_points(event):
        state['X_user_class0'].clear()
        state['X_user_class1'].clear()
        with output:
            output.clear_output(wait=True)
            print("🗑️ Cleared all user-added points")
        train_and_plot()

    # Connect buttons
    train_button.on_click(train_and_plot)
    clear_button.on_click(clear_user_points)
    add_class0_button.on_click(add_class0_point)
    add_class1_button.on_click(add_class1_point)

    # Layout widgets
    controls_row1 = HBox([epoch_field, lr_field])
    controls_row2 = HBox([network_field])
    buttons_row = HBox([train_button, clear_button])
    
    add_point_label = widgets.HTML(value="<b>Add Point:</b>")
    add_point_row = HBox([add_point_label, add_x_field, add_y_field, add_class0_button, add_class1_button])
    
    ui = VBox([
        controls_row1, 
        controls_row2, 
        buttons_row, 
        add_point_row, 
        output, 
        plot_output
    ])
    
    # Display interface
    display(ui)

    # Print instructions
    print("\n" + "="*70)
    print("🎯 INTERACTIVE MLP VISUALIZATION - GOOGLE COLAB")
    print("="*70)
    print("📝 HOW TO ADD POINTS:")
    print("   1. Enter X and Y coordinates in the fields")
    print("   2. Click 'Add Class 0' (red circles) or 'Add Class 1' (blue X's)")
    print("   3. Click 'TRAIN' button to retrain the model with new points")
    print("")
    print("⚙️  EXPERIMENT:")
    print("   - Change 'Epochs' for more/less training")
    print("   - Adjust 'Learning Rate' (try 0.0001 to 0.01)")
    print("   - Modify 'Network Size' (e.g., '10 10' or '50 50 50')")
    print("")
    print("🗑️  Click 'CLEAR ADDED POINTS' to reset and start over")
    print("="*70 + "\n")

    # Initial training and plot
    train_and_plot()


def create_interactive_mlp_visualization_with_config(X, y, title, config_overrides=None):
    """
    Create an interactive MLP visualization using configuration from Config.yml.
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
