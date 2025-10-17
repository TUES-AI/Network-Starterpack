#!/usr/bin/env python3
"""
Example usage of per-pixel tinting functionality
"""

import numpy as np
from responsive_plot import initialize_plot, add_dot, set_tint, set_pixel_tint, set_region_tint, remove_tint

# Initialize the plot with higher resolution for better pixel control
initialize_plot()

print("Plot initialized! Adding some dots...")

# Add some dots
add_dot(0.5, 0.5, color='red')
add_dot(-0.3, 0.7, color='blue')
add_dot(0.8, -0.2, color='green')

print("Dots added. Now demonstrating per-pixel tinting...")

# Example 1: Set individual pixel tints
print("\n1. Setting individual pixel tints...")
set_pixel_tint(0.0, 0.0, (1.0, 0.0, 0.0, 0.5))  # Red at center
set_pixel_tint(0.5, 0.5, (0.0, 1.0, 0.0, 0.5))  # Green at (0.5, 0.5)
set_pixel_tint(-0.5, -0.5, (0.0, 0.0, 1.0, 0.5))  # Blue at (-0.5, -0.5)

# Example 2: Set region tints
print("\n2. Setting region tints...")
set_region_tint((-0.2, 0.2), (-0.2, 0.2), (1.0, 1.0, 0.0, 0.3))  # Yellow square
set_region_tint((0.6, 0.9), (0.6, 0.9), (1.0, 0.0, 1.0, 0.4))  # Magenta square

# Example 3: Create a gradient using pixel array
print("\n3. Creating gradient using pixel array...")
resolution = 100
# Create a gradient from blue to red
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
X, Y = np.meshgrid(x, y)

# Create RGBA array: red increases with x, blue decreases with x
pixel_array = np.zeros((resolution, resolution, 4))
pixel_array[:, :, 0] = (X + 1) / 2  # Red channel
pixel_array[:, :, 2] = 1 - (X + 1) / 2  # Blue channel
pixel_array[:, :, 1] = 0.2  # Some green
pixel_array[:, :, 3] = 0.3  # Alpha (transparency)

set_tint(pixel_array=pixel_array)

print("\nPer-pixel tinting complete!")
print("You can see:")
print("- Individual colored pixels at specific coordinates")
print("- Colored regions (squares)")
print("- A smooth gradient from blue to red")
print("\nAll dots remain visible on top of the tint.")

# Keep the plot open
print("\nPlot will stay open. You can continue to add/remove dots and tints.")
print("When done, close the plot window manually or call close_plot().")

while True:
    pass
