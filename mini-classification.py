import sklearn.datasets
import numpy as np
import random

def create_ascii_art_classification(sample_data, target):
    """Convert 3D classification data to ASCII art visualization"""
    # Normalize the 3D data to fit in a 2D grid
    # We'll use the first two features for x,y positions and the third for intensity
    x_norm = int((sample_data[0] + 3) / 6 * 15)  # Scale to 0-15 assuming data in [-3,3]
    y_norm = int((sample_data[1] + 3) / 6 * 15)  # Scale to 0-15
    intensity = int((sample_data[2] + 3) / 6 * 5)  # Scale to 0-5 for ASCII chars

    # Define ASCII characters for different intensity levels
    ascii_chars = [' ', '.', ':', 'o', 'O', '@']

    # Create a 16x16 grid
    grid = [[' ' for _ in range(16)] for _ in range(16)]

    # Place the point on the grid
    if 0 <= x_norm < 16 and 0 <= y_norm < 16:
        grid[y_norm][x_norm] = ascii_chars[intensity]

    # Convert grid to ASCII art
    result = []
    for row in grid:
        line = ''.join(row)
        result.append(line)

    return result

# Generate the classification dataset
X, y = sklearn.datasets.make_classification(
    n_features=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Select a random sample
random_index = random.randint(0, len(X) - 1)
sample_data = X[random_index]
sample_label = y[random_index]

# Create ASCII art
ascii_art = create_ascii_art_classification(sample_data, sample_label)

# Print the visualization
print(f"Random Classification Sample #{random_index}")
print(f"Features: {[f'{x:.2f}' for x in sample_data]}")
print(f"Class Label: {sample_label}")
print("\n2D Projection Visualization (features 1 & 2):")
print("+" + "-" * 16 + "+")
for line in ascii_art:
    print(f"|{line}|")
print("+" + "-" * 16 + "+")

# Additional information
print(f"\nFeature 3 (intensity): {sample_data[2]:.2f}")
print("Coordinate mapping:")
print(f"  Feature 1 (x-axis): {sample_data[0]:.2f} → position {int((sample_data[0] + 3) / 6 * 15)}")
print(f"  Feature 2 (y-axis): {sample_data[1]:.2f} → position {int((sample_data[1] + 3) / 6 * 15)}")
print(f"  Feature 3 (intensity): {sample_data[2]:.2f} → character '{ascii_art[int((sample_data[1] + 3) / 6 * 15)][int((sample_data[0] + 3) / 6 * 15)]}'")
