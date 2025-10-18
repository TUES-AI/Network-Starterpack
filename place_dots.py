import responsive_plot as plot
import random

plot.initialize_plot()

plot.add_dot(0.5, 0.5, color='red')

def random_point():
    return random.random()*2-1

while True:
    plot.add_dot(random_point(), random_point(), color='blue')
