import sklearn.datasets
import numpy as np
import random

def create_ascii_art_digit(digit_data):
    """Convert 8x8 digit image to ASCII art"""
    # Define ASCII characters for different intensity levels
    ascii_chars = [' ', '.', ':', 'o', 'O', '@', '#']

    result = []
    for row in digit_data:
        line = ''
        for pixel in row:
            # Map pixel value (0-16) to ASCII character
            char_index = min(int(pixel / 3), len(ascii_chars) - 1)
            line += ascii_chars[char_index]
        result.append(line)

    return result

# Load the digits dataset
digits = sklearn.datasets.load_digits()

# Select a random digit
random_index = random.randint(0, len(digits.data) - 1)
digit_image = digits.images[random_index]
digit_label = digits.target[random_index]

# Create ASCII art
ascii_art = create_ascii_art_digit(digit_image)

# Print the visualization
print(f"Random Digit #{random_index}")
print(f"Actual Label: {digit_label}")
print("\nASCII Art Representation:")
print("+" + "-" * 8 + "+")
for line in ascii_art:
    print(f"|{line}|")
print("+" + "-" * 8 + "+")

# Also show the raw pixel values
print("\nRaw Pixel Values (0-16):")
for i, row in enumerate(digit_image):
    print(f"Row {i}: {[int(x) for x in row]}")
