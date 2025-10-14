import sklearn.datasets
import numpy as np
import random

def create_ascii_art_moons(sample_data, target):
    """Convert 2D moons data to ASCII art visualization"""
    # Scale the 2D data to fit in a 16x16 grid
    # Moons data typically ranges around [-1.5, 2.5] for x and [-1, 1.5] for y
    x_norm = int((sample_data[0] + 2) / 4 * 15)  # Scale to 0-15
    y_norm = int((sample_data[1] + 1.5) / 3 * 15)  # Scale to 0-15

    # Define ASCII characters for different classes
    class_chars = ['O', 'X']  # Different characters for each class

    # Create a 16x16 grid
    grid = [[' ' for _ in range(16)] for _ in range(16)]

    # Place the point on the grid
    if 0 <= x_norm < 16 and 0 <= y_norm < 16:
        grid[y_norm][x_norm] = class_chars[target]

    # Convert grid to ASCII art
    result = []
    for row in grid:
        line = ''.join(row)
        result.append(line)

    return result

# Generate the moons dataset
X, y = sklearn.datasets.make_moons(noise=0.2, random_state=42)

# Select a random sample
random_index = random.randint(0, len(X) - 1)
sample_data = X[random_index]
sample_label = y[random_index]

# Create ASCII art
ascii_art = create_ascii_art_moons(sample_data, sample_label)

# Print the visualization
print(f"Random Moons Sample #{random_index}")
print(f"Coordinates: ({sample_data[0]:.2f}, {sample_data[1]:.2f})")
print(f"Class Label: {sample_label}")
print("\n2D Visualization:")
print("+" + "-" * 16 + "+")
for line in ascii_art:
    print(f"|{line}|")
print("+" + "-" * 16 + "+")

# Additional information
print("\nLegend:")
print("  'O' = Class 0 (first moon)")
print("  'X' = Class 1 (second moon)")
print(f"\nCoordinate mapping:")
print(f"  X: {sample_data[0]:.2f} → position {int((sample_data[0] + 2) / 4 * 15)}")
print(f"  Y: {sample_data[1]:.2f} → position {int((sample_data[1] + 1.5) / 3 * 15)}")
print(f"  Character: '{ascii_art[int((sample_data[1] + 1.5) / 3 * 15)][int((sample_data[0] + 2) / 4 * 15)]}'")

# Show a few more samples to demonstrate the moon shapes
print("\n--- Additional Samples ---")
for i in range(3):
    idx = random.randint(0, len(X) - 1)
    coords = X[idx]
    label = y[idx]
    x_pos = int((coords[0] + 2) / 4 * 15)
    y_pos = int((coords[1] + 1.5) / 3 * 15)
    print(f"Sample {idx}: ({coords[0]:.2f}, {coords[1]:.2f}) → Class {label} at ({x_pos}, {y_pos})")
