from responsive_plot import initialize_plot, add_dot
import random

initialize_plot()

add_dot(0.5, 0.5, color='red')

print("Plot is now responsive! You can add/remove dots dynamically.")

while True:
    add_dot(random.random(), random.random(), color='blue')
