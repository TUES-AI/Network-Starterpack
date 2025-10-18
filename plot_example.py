#!/usr/bin/env python3
"""
Example usage of the responsive plotting system
"""

from responsive_plot import initialize_plot, add_dot, remove_dot, clear_dots, get_dots, close_plot

# Initialize the plot at the beginning of your file
initialize_plot()

print("Plot initialized! You can now add and remove dots dynamically.")
print("The plot will stay open and responsive.")

# Example: Add some dots
print("\nAdding dots...")
add_dot(0.5, 0.5, color='red')
add_dot(-0.3, 0.7, color='blue')
add_dot(0.8, -0.2, color='green')

print(f"Current dots: {get_dots()}")

# Example: Remove a dot by coordinates
print("\nRemoving dot at (0.5, 0.5)...")
remove_dot(x=0.5, y=0.5)

print(f"Current dots after removal: {get_dots()}")

# Example: Add more dots
print("\nAdding more dots...")
add_dot(0.2, 0.3, color='purple')
add_dot(-0.6, -0.4, color='orange')

print(f"Current dots: {get_dots()}")

# Example: Clear all dots
print("\nClearing all dots...")
clear_dots()

print(f"Current dots after clearing: {get_dots()}")

print("\nYou can continue to call add_dot(), remove_dot(), etc. without restarting the plot!")
print("When done, call close_plot() to close the window.")

# Keep the script running to see the plot
input("\nPress Enter to close the plot...")

# Close the plot when done
close_plot()
print("Plot closed.")