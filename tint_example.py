#!/usr/bin/env python3
"""
Example usage of the background tint functionality
"""

from responsive_plot import initialize_plot, add_dot, set_tint, remove_tint

# Initialize the plot
initialize_plot()

print("Plot initialized! Adding some dots...")

# Add some dots
add_dot(0.5, 0.5, color='red')
add_dot(-0.3, 0.7, color='blue')
add_dot(0.8, -0.2, color='green')

print("Dots added. Now adding a light blue tint...")

# Add a light blue tint with 40% opacity
set_tint(color=(0.8, 0.9, 1.0, 0.4))  # Light blue

print("Tint added! The background should be light blue and transparent.")
print("The dots should appear on top of the tint.")

# Wait a bit, then change to a different tint
import time
time.sleep(2)

print("Changing to a warm peach tint...")
set_tint(color=(1.0, 0.95, 0.9, 0.3))  # Warm peach

# Wait a bit, then remove the tint
time.sleep(2)

print("Removing the tint...")
remove_tint()

print("Tint removed! Plot should be back to normal.")

# Keep the plot open
print("\nPlot will stay open. You can continue to add/remove dots and tints.")
print("When done, close the plot window manually or call close_plot().")