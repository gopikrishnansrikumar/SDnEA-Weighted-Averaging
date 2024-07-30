import math
import numpy as np


def calculate_weights(pd1, pd2):
    if pd1 == 1.0:
        pd1 = 0.99
    if pd2 == 1.0:
        pd2 = 0.99
    if pd1 == 0.0:
        pd1 = 0.001
    if pd2 == 0.0:
        pd2 = 0.001
    mean = (pd1 + pd2) / 2

    # Calculate the sum of squares of differences
    sum_of_squares = (pd1 - mean) ** 2 + (pd2 - mean) ** 2

    # Calculate the standard deviation
    std_dev = math.sqrt(sum_of_squares / 2)

    if std_dev == 0:
        std_dev = 1

    # Calculate w1
    w1 = (1 + (pd1 * math.log2(pd1)) + ((1 - pd1) * math.log2(1 - pd1))) / std_dev
    # w1 = (1 + pd1 * np.log2(pd1) + (1 - pd1) * np.log2(1 - pd1)) / std_dev

    # Calculate w2
    w2 = (1 + (pd2 * math.log2(pd2)) + ((1 - pd2) * math.log2(1 - pd2))) / std_dev

    return w1, w2

if __name__ == '__main__':
    pd1 = 0.2
    pd2 = 0.98
    w1, w2 = calculate_weights(pd1, pd2)
    print(w1, w2)