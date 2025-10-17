import sklearn.datasets
import numpy as np
import random

def create_ascii_art_digit(digit_data):
    """Convert 8x8 digit image to ASCII art"""
    ascii_chars = [' ', '.', ':', 'o', 'O', '@', '#']

    result = []
    for row in digit_data:
        line = ''
        for pixel in row:
            char_index = min(int(pixel / 3), len(ascii_chars) - 1)
            line += ascii_chars[char_index]
        result.append(line)

    return result

digits = sklearn.datasets.load_digits()

random_index = random.randint(0, len(digits.data) - 1)
digit_image = digits.images[random_index]
digit_label = digits.target[random_index]

ascii_art = create_ascii_art_digit(digit_image)

print(f"Actual Label: {digit_label}")
print("+" + "-" * 8 + "+")
for line in ascii_art:
    print(f"|{line}|")
print("+" + "-" * 8 + "+")
