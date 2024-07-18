import numpy as np
from superquadrics.superquadrics import SuperQuadrics

# Helper functions
# Helper functions
def convert_to_polydata(superquadric):
    points = np.vstack((superquadric.x.flatten(), superquadric.y.flatten(), superquadric.z.flatten())).T
    num_points = superquadric.x.shape[0]
    
    polys = []
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            p1 = i * num_points + j
            p2 = p1 + 1
            p3 = p1 + num_points
            p4 = p3 + 1
            polys.extend([4, p1, p2, p4, p3])  # A quad made of 4 points
    
    return points.flatten().tolist(), polys

def create_point_cloud(num_points=1000, shape="sphere"):
    if shape == "sphere":
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
    return np.vstack((x, y, z)).T

def create_superquadric(size, shape):
    superquadric = SuperQuadrics(size=size, shape=shape)
    return convert_to_polydata(superquadric)
