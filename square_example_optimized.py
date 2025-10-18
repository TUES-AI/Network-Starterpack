"""
Optimized example using batch square adding for much better performance
"""

import numpy as np
from responsive_plot import initialize_plot, add_dot, add_squares_batch

initialize_plot()

print("Plot initialized! Creating transparent background with optimized batch adding...")

print("\nCreating dense grid of transparent squares using batch method...")

resolution = 40

square_size = 2.0 / resolution

squares_data = []

for i in range(resolution):
    for j in range(resolution):
        x = -1.0 + (j * square_size) + (square_size / 2)
        y = -1.0 + (i * square_size) + (square_size / 2)

        red = i / (resolution - 1)
        green = j / (resolution - 1)
        blue = 0.5
        alpha = 0.5

        color = (red, green, blue, alpha)

        squares_data.append((x, y, square_size, color))

print(f"Adding {len(squares_data)} squares in one batch...")
add_squares_batch(squares_data)

print("Squares added! This should be much faster than individual adding.")
print("Squares completely fill the plot area with no gaps")
print("All squares are 50% transparent")

print("\nAdding dots on top of the transparent background...")
add_dot(0.5, 0.5, color='white', markersize=12)
add_dot(-0.3, 0.7, color='yellow', markersize=10)
add_dot(0.8, -0.2, color='cyan', markersize=10)
add_dot(-0.6, -0.4, color='magenta', markersize=10)

print("Dots added on top - they remain visible through the transparent squares")

print("\nThe entire plot area is now filled with transparent squares!")
print("No gaps between squares, and dots remain visible on top.")
print("Performance should be significantly better with batch adding.")

print("\nPlot will stay open. You can continue to experiment with different patterns.")
print("When done, close the plot window manually or call close_plot().")
