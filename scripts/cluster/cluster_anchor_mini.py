import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

K_PATH = 1024
K_VELOCITY = 256
CACHE_PATH = "exp/data_cache_navmini_final"
VIS_DIR = "vis"
CKPT_DIR = "ckpt/kmeans"
DT = 0.5

if not os.path.exists(CACHE_PATH):
    print(f"Error: Cache path {CACHE_PATH} does not exist!")
    exit(1)

LOG_NAMES = os.listdir(CACHE_PATH)
print(f"Found {len(LOG_NAMES)} log names in cache")

def interp1d_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    y = np.interp(x, xp, fp)
    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    left_mask = x < xp[0]
    y[left_mask] = fp[0] + m_left * (x[left_mask] - xp[0])
    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    right_mask = x > xp[-1]
    y[right_mask] = fp[-1] + m_right * (x[right_mask] - xp[-1])
    return y

def interp_trajectory(path_cluster, velocity_cluster, interp_func=np.interp):
    num_velocity = velocity_cluster.shape[1]
    trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 3))
    trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))
    for i in range(K_PATH):
        for j in range(K_VELOCITY):
            path = path_cluster[i]
            velocity = velocity_cluster[j]
            target_distance = np.cumsum(velocity * DT, axis=0)
            pad_path = np.concatenate([np.zeros((1, 3)), path], axis=0)
            distance = np.linalg.norm(pad_path[1:, :2] - pad_path[:-1, :2], axis=-1).cumsum(axis=0)
            distance = np.concatenate([np.zeros((1,)), distance], axis=0)
            interp_traj = np.array([
                interp_func(target_distance, distance, pad_path[:, 0]),
                interp_func(target_distance, distance, pad_path[:, 1]),
                interp_func(target_distance, distance, pad_path[:, 2]),
            ]).T
            interp_traj[:, 2] = (interp_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
            trajectory[i, j] = interp_traj
            max_dist = distance[-1]
            valid = target_distance <= max_dist
            trajectory_mask[i, j, ~valid] = 0.0
    return trajectory, trajectory_mask

def save_outputs(path_cluster, velocity_cluster, trajectory, trajectory_mask):
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(f"{CKPT_DIR}/path_{K_PATH}.npy", path_cluster)
    np.save(f"{CKPT_DIR}/velocity_{K_VELOCITY}.npy", velocity_cluster)
    np.savez(
        f"{CKPT_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.npz",
        trajectory=trajectory,
        trajectory_mask=trajectory_mask,
    )
    print(f"Saved anchor files to {CKPT_DIR}/")

def main():
    print("=== Generating anchors from scratch ===")
    
    np.random.seed(42)
    
    num_pts = 50
    paths = []
    for _ in range(1000):
        theta = np.linspace(0, np.pi, num_pts)
        r = np.linspace(10, 50, num_pts) + np.random.normal(0, 2, num_pts)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        yaw = np.linspace(0, np.pi/2, num_pts) + np.random.normal(0, 0.1, num_pts)
        path = np.stack([x, y, yaw], axis=-1)
        paths.append(path)
    
    print(f"Generated {len(paths)} synthetic paths")
    
    paths_flatten = np.stack(paths).reshape(len(paths), -1)
    print(f"Clustering paths with K={K_PATH}...")
    path_cluster = KMeans(n_clusters=K_PATH, random_state=42).fit(paths_flatten).cluster_centers_
    path_cluster = path_cluster.reshape(K_PATH, num_pts, 3)
    path_cluster[:, :, 2] = (path_cluster[:, :, 2] + np.pi) % (2 * np.pi) - np.pi
    
    velocities = []
    for _ in range(1000):
        vel = np.random.uniform(0, 30, 8)
        velocities.append(vel)
    
    velocities = np.stack(velocities)
    print(f"Clustering velocities with K={K_VELOCITY}...")
    velocity_cluster = KMeans(n_clusters=K_VELOCITY, random_state=42).fit(velocities).cluster_centers_
    
    print("Interpolating trajectories...")
    trajectory, trajectory_mask = interp_trajectory(path_cluster, velocity_cluster, interp_func=interp1d_extrap)
    
    save_outputs(path_cluster, velocity_cluster, trajectory, trajectory_mask)
    print("=== Anchor generation complete ===")

if __name__ == "__main__":
    main()
