import numpy as np
from responsive_plot import initialize_plot, add_dot, add_square, clear_squares

initialize_plot()

resolution = 40

square_size = 2.0 / resolution

background_array = np.zeros((resolution, resolution, 4))

for i in range(resolution):
    for j in range(resolution):
        red = i / (resolution - 1)
        green = j / (resolution - 1)
        blue = 0.5
        alpha = 0.5

        background_array[i, j] = [red, green, blue, alpha]

for i in range(resolution):
    for j in range(resolution):
        x = -1.0 + (j * square_size) + (square_size / 2)
        y = -1.0 + (i * square_size) + (square_size / 2)

        color = background_array[i, j]

        add_square(x, y, square_size, color)

add_dot(0.5, 0.5, color='white', markersize=12)
add_dot(-0.3, 0.7, color='yellow', markersize=10)
add_dot(0.8, -0.2, color='cyan', markersize=10)
add_dot(-0.6, -0.4, color='magenta', markersize=10)

clear_squares()

for i in range(resolution):
    for j in range(resolution):
        x = -1.0 + (j * square_size) + (square_size / 2)
        y = -1.0 + (i * square_size) + (square_size / 2)

        alpha = 0.2 + 0.6 * (i / resolution)

        if (i + j) % 3 == 0:
            color = (0.8, 0.2, 0.2, alpha)
        elif (i + j) % 3 == 1:
            color = (0.2, 0.8, 0.2, alpha)
        else:
            color = (0.2, 0.2, 0.8, alpha)

        add_square(x, y, square_size, color)
