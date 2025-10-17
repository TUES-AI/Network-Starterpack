import matplotlib.pyplot as plt
import numpy as np

class ResponsivePlot:
    def __init__(self, figsize=(5, 5), dpi=120, resolution=100):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.dots = []  # Store dot objects
        self.dot_positions = []  # Store dot coordinates
        self.background_patch = None  # Store background tint object
        self.background_image = None  # Store background image object
        self.resolution = resolution  # Resolution for per-pixel tinting

        # Initialize the plot with 8 roles (axes)
        self._setup_plot()

        # Enable interactive mode
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
        return len(self.dots) - 1  # Return dot index

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

    def set_tint(self, color=None, pixel_array=None):
        """Set a background tint - either uniform color or per-pixel array

        Args:
            color: RGB or RGBA tuple for uniform tint, e.g., (1.0, 0.9, 0.8, 0.3)
            pixel_array: 2D array of RGBA values for per-pixel tinting
        """
        # Remove existing background if any
        self._remove_background()

        if pixel_array is not None:
            # Per-pixel tinting using image
            self._set_pixel_tint(pixel_array)
        elif color is not None:
            # Uniform tint using rectangle
            self._set_uniform_tint(color)

    def _set_uniform_tint(self, color):
        """Set a uniform background tint"""
        self.background_patch = plt.Rectangle(
            (-1, -1),  # bottom left corner
            2, 2,      # width and height
            facecolor=color,
            zorder=-1  # Ensure it's behind everything
        )
        self.ax.add_patch(self.background_patch)
        self._update_plot()

    def _set_pixel_tint(self, pixel_array):
        """Set per-pixel tint using an image array"""
        # Ensure the array has the right shape
        if pixel_array.ndim == 2:
            # Grayscale to RGB
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
            # RGB, add alpha channel
            pixel_array = np.dstack([pixel_array, np.ones(pixel_array.shape[:2])])
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 4:
            # Already RGBA, use as is
            pass
        else:
            raise ValueError("pixel_array must be 2D (grayscale), 3D (RGB), or 4D (RGBA)")

        # Create image
        self.background_image = self.ax.imshow(
            pixel_array,
            extent=[-1, 1, -1, 1],  # Cover entire plot area
            origin='lower',
            zorder=-1,  # Behind everything
            interpolation='nearest'  # Sharp pixel boundaries
        )
        self._update_plot()

    def set_pixel_tint(self, x, y, color):
        """Set tint for a specific pixel coordinate

        Args:
            x, y: Coordinate in plot space (-1 to 1)
            color: RGBA tuple for the pixel
        """
        # Convert coordinate to pixel index
        i = int((y + 1) / 2 * (self.resolution - 1))
        j = int((x + 1) / 2 * (self.resolution - 1))

        # Ensure we have a pixel array
        if self.background_image is None:
            # Create initial transparent array
            pixel_array = np.zeros((self.resolution, self.resolution, 4))
            pixel_array[:, :, 3] = 0  # Fully transparent
            # Create the image
            self.background_image = self.ax.imshow(
                pixel_array,
                extent=[-1, 1, -1, 1],
                origin='lower',
                zorder=-1,
                interpolation='nearest'
            )
        else:
            # Get current array
            pixel_array = self.background_image.get_array()

        # Set the pixel color
        pixel_array[i, j] = color

        # Update the image
        self.background_image.set_array(pixel_array)
        self._update_plot()

    def set_region_tint(self, x_range, y_range, color):
        """Set tint for a rectangular region

        Args:
            x_range: (x_min, x_max) in plot coordinates
            y_range: (y_min, y_max) in plot coordinates
            color: RGBA tuple for the region
        """
        # Convert coordinates to pixel indices
        i_min = int((y_range[0] + 1) / 2 * (self.resolution - 1))
        i_max = int((y_range[1] + 1) / 2 * (self.resolution - 1))
        j_min = int((x_range[0] + 1) / 2 * (self.resolution - 1))
        j_max = int((x_range[1] + 1) / 2 * (self.resolution - 1))

        # Ensure we have a pixel array
        if self.background_image is None:
            pixel_array = np.zeros((self.resolution, self.resolution, 4))
            pixel_array[:, :, 3] = 0
            # Create the image
            self.background_image = self.ax.imshow(
                pixel_array,
                extent=[-1, 1, -1, 1],
                origin='lower',
                zorder=-1,
                interpolation='nearest'
            )
        else:
            pixel_array = self.background_image.get_array()

        # Set the region color
        pixel_array[i_min:i_max+1, j_min:j_max+1] = color

        # Update the image
        self.background_image.set_array(pixel_array)
        self._update_plot()

    def _remove_background(self):
        """Remove any existing background"""
        if self.background_patch is not None:
            self.background_patch.remove()
            self.background_patch = None
        if self.background_image is not None:
            self.background_image.remove()
            self.background_image = None

    def remove_tint(self):
        """Remove the background tint"""
        self._remove_background()
        self._update_plot()

    def _update_plot(self):
        """Update the plot display"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot"""
        plt.ioff()
        plt.close(self.fig)

# Global plot instance
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

def set_tint(color=None, pixel_array=None):
    """Set a background tint - either uniform color or per-pixel array

    Args:
        color: RGB or RGBA tuple for uniform tint, e.g., (1.0, 0.9, 0.8, 0.3)
        pixel_array: 2D array of RGBA values for per-pixel tinting
    """
    if plot is None:
        print("Plot not initialized")
        return
    plot.set_tint(color=color, pixel_array=pixel_array)

def set_pixel_tint(x, y, color):
    """Set tint for a specific pixel coordinate

    Args:
        x, y: Coordinate in plot space (-1 to 1)
        color: RGBA tuple for the pixel
    """
    if plot is None:
        print("Plot not initialized")
        return
    plot.set_pixel_tint(x, y, color)

def set_region_tint(x_range, y_range, color):
    """Set tint for a rectangular region

    Args:
        x_range: (x_min, x_max) in plot coordinates
        y_range: (y_min, y_max) in plot coordinates
        color: RGBA tuple for the region
    """
    if plot is None:
        print("Plot not initialized")
        return
    plot.set_region_tint(x_range, y_range, color)

def remove_tint():
    """Remove the background tint"""
    if plot is None:
        print("Plot not initialized")
        return
    plot.remove_tint()

def close_plot():
    """Close the plot"""
    global plot
    if plot is not None:
        plot.close()
        plot = None