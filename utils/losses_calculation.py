from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_iou(point_cloud, superquadric_points):
    pc_min, pc_max = point_cloud.min(axis=0), point_cloud.max(axis=0)
    sq_min, sq_max = superquadric_points.min(axis=0), superquadric_points.max(axis=0)
    
    inter_min = np.maximum(pc_min, sq_min)
    inter_max = np.minimum(pc_max, sq_max)
    
    if np.any(inter_min >= inter_max):
        return 0.0
    
    inter_volume = np.prod(inter_max - inter_min)
    pc_volume = np.prod(pc_max - pc_min)
    sq_volume = np.prod(sq_max - sq_min)
    
    union_volume = pc_volume + sq_volume - inter_volume
    iou = inter_volume / union_volume
    return iou

def calculate_chamfer_distance(pc1, pc2):
    dists_pc1_to_pc2 = pairwise_distances(pc1, pc2).min(axis=1)
    dists_pc2_to_pc1 = pairwise_distances(pc2, pc1).min(axis=1)
    return np.mean(dists_pc1_to_pc2) + np.mean(dists_pc2_to_pc1)