import responsive_plot  as plot

plot.initialize_plot()

resolution = 20
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

plot.add_squares_timed(squares_data, duration=1.0)
