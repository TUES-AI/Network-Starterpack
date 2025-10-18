import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.collections import PatchCollection

class ResponsivePlot:
    def __init__(self, figsize=(5, 5), dpi=120):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.dots = []
        self.dot_positions = []
        self.squares = []
        self.square_collection = None
        self.square_data = []

        self._setup_plot()

        plt.ion()
        plt.show()

    def _setup_plot(self):
        """Initialize the plot with coordinate system"""
        self.ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        self.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        self.ax.grid(alpha=0.3)
        self.fig.tight_layout(pad=0)

    def add_dot(self, x, y, color='red', markersize=8):
        """Add a dot to the plot"""
        dot = self.ax.plot(x, y, 'o', color=color, markersize=markersize)[0]
        self.dots.append(dot)
        self.dot_positions.append((x, y))
        self._update_plot()
        return len(self.dots) - 1

    def remove_dot(self, index=None, x=None, y=None):
        """Remove a dot by index or by coordinates"""
        if index is not None:
            if 0 <= index < len(self.dots):
                self.dots[index].remove()
                self.dots.pop(index)
                self.dot_positions.pop(index)
                self._update_plot()
                return True
        elif x is not None and y is not None:
            for i, (dot_x, dot_y) in enumerate(self.dot_positions):
                if abs(dot_x - x) < 0.01 and abs(dot_y - y) < 0.01:
                    self.dots[i].remove()
                    self.dots.pop(i)
                    self.dot_positions.pop(i)
                    self._update_plot()
                    return True
        return False

    def clear_all_dots(self):
        """Remove all dots from the plot"""
        for dot in self.dots:
            dot.remove()
        self.dots.clear()
        self.dot_positions.clear()
        self._update_plot()

    def get_dot_positions(self):
        """Get all current dot positions"""
        return self.dot_positions.copy()

    def add_square(self, x, y, size, color=(1.0, 0.0, 0.0, 1.0)):
        """Add a square with no borders and RGBA color

        Args:
            x, y: Center coordinates of the square
            size: Side length of the square
            color: RGBA tuple (red, green, blue, alpha) where each value is 0.0 to 1.0
        """
        self.square_data.append((x, y, size, color))

        self._rebuild_squares()

        return len(self.square_data) - 1

    def add_squares_batch(self, squares_data):
        """Add multiple squares at once for better performance

        Args:
            squares_data: List of tuples (x, y, size, color)
        """
        self.square_data.extend(squares_data)
        self._rebuild_squares()

    def add_squares_timed(self, squares_data, duration=3.0):
        """Add multiple squares gradually over specified duration

        Args:
            squares_data: List of tuples (x, y, size, color)
            duration: Time in seconds over which to spread the placement
        """
        if not squares_data:
            return

        total_squares = len(squares_data)

        target_updates_per_second = 30
        total_updates = int(duration * target_updates_per_second)

        total_updates = max(total_updates, 10)
        total_updates = min(total_updates, total_squares)

        batch_size = max(1, total_squares // total_updates)

        time_interval = duration / (total_squares / batch_size)


        for i in range(0, total_squares, batch_size):
            batch = squares_data[i:i + batch_size]
            self.square_data.extend(batch)
            self._rebuild_squares()

            time.sleep(time_interval)

    def remove_square(self, index=None):
        """Remove a square by index"""
        if index is not None and 0 <= index < len(self.square_data):
            self.square_data.pop(index)
            self._rebuild_squares()
            return True
        return False

    def clear_all_squares(self):
        """Remove all squares from the plot"""
        self.square_data.clear()
        self._rebuild_squares()

    def _rebuild_squares(self):
        """Rebuild the entire square collection for optimal performance"""
        if self.square_collection is not None:
            self.square_collection.remove()

        if not self.square_data:
            self.square_collection = None
            self._update_plot()
            return

        patches = []
        colors = []

        for x, y, size, color in self.square_data:
            half_size = size / 2
            bottom_left_x = x - half_size
            bottom_left_y = y - half_size

            patch = plt.Rectangle(
                (bottom_left_x, bottom_left_y),
                size, size
            )
            patches.append(patch)
            colors.append(color)

        self.square_collection = PatchCollection(
            patches,
            facecolors=colors,
            edgecolors='none',
            match_original=True
        )

        self.square_collection.set_zorder(-1)
        self.ax.add_collection(self.square_collection)

        self._update_plot()

    def _update_plot(self):
        """Update the plot display"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot"""
        plt.ioff()
        plt.close(self.fig)

plot = None

def initialize_plot():
    """Initialize the responsive plot - call this at the beginning of your file"""
    global plot
    plot = ResponsivePlot()
    return plot

def add_dot(x, y, color='red', markersize=8):
    """Add a dot to the plot"""
    if plot is None:
        initialize_plot()
    return plot.add_dot(x, y, color, markersize)

def remove_dot(index=None, x=None, y=None):
    """Remove a dot from the plot"""
    if plot is None:
        print("Plot not initialized")
        return False
    return plot.remove_dot(index, x, y)

def clear_dots():
    """Clear all dots from the plot"""
    if plot is None:
        print("Plot not initialized")
        return
    plot.clear_all_dots()

def get_dots():
    """Get all current dot positions"""
    if plot is None:
        return []
    return plot.get_dot_positions()

def add_square(x, y, size, color=(1.0, 0.0, 0.0, 1.0)):
    """Add a square with no borders and RGBA color

    Args:
        x, y: Center coordinates of the square
        size: Side length of the square
        color: RGBA tuple (red, green, blue, alpha) where each value is 0.0 to 1.0
    """
    if plot is None:
        print("Plot not initialized")
        return
    return plot.add_square(x, y, size, color)

def add_squares_batch(squares_data):
    """Add multiple squares at once for better performance

    Args:
        squares_data: List of tuples (x, y, size, color)
    """
    if plot is None:
        print("Plot not initialized")
        return
    plot.add_squares_batch(squares_data)

def add_squares_timed(squares_data, duration=3.0):
    """Add multiple squares gradually over specified duration

    Args:
        squares_data: List of tuples (x, y, size, color)
        duration: Time in seconds over which to spread the placement
    """
    if plot is None:
        print("Plot not initialized")
        return
    plot.add_squares_timed(squares_data, duration)

def remove_square(index=None):
    """Remove a square by index"""
    if plot is None:
        print("Plot not initialized")
        return False
    return plot.remove_square(index)

def clear_squares():
    """Remove all squares from the plot"""
    if plot is None:
        print("Plot not initialized")
        return
    plot.clear_all_squares()

def close_plot():
    """Close the plot"""
    global plot
    if plot is not None:
        plot.close()
        plot = None
