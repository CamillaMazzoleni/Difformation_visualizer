import numpy as np  


# Define SuperQuadrics and helper functions
def fexp(x, e):
    return np.sign(x) * np.abs(x) ** e