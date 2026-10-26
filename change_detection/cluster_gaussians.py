from __future__ import annotations

from collections import defaultdict

import numpy as np


def _voxel_key(point, eps):
    return tuple(np.floor(point / eps).astype(np.int64).tolist())


def _region_query(xyz, point_idx, eps, grid):
    base = np.floor(xyz[point_idx] / eps).astype(np.int64)
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                key = (base[0] + dx, base[1] + dy, base[2] + dz)
                for other in grid.get(key, []):
                    if np.linalg.norm(xyz[other] - xyz[point_idx]) <= eps:
                        neighbors.append(other)
    return neighbors


def dbscan_numpy(xyz, eps, min_samples):
    n_points = xyz.shape[0]
    labels = np.full(n_points, -1, dtype=np.int64)
    if n_points == 0:
        return labels

    grid = defaultdict(list)
    for idx, point in enumerate(xyz):
        grid[_voxel_key(point, eps)].append(idx)

    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0
    for idx in range(n_points):
        if visited[idx]:
            continue
        visited[idx] = True
        neighbors = _region_query(xyz, idx, eps, grid)
        if len(neighbors) < min_samples:
            continue

        labels[idx] = cluster_id
        seeds = list(neighbors)
        seed_set = set(seeds)
        seed_pos = 0
        while seed_pos < len(seeds):
            current = seeds[seed_pos]
            if not visited[current]:
                visited[current] = True
                current_neighbors = _region_query(xyz, current, eps, grid)
                if len(current_neighbors) >= min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in seed_set:
                            seeds.append(neighbor)
                            seed_set.add(neighbor)
            if labels[current] < 0:
                labels[current] = cluster_id
            seed_pos += 1
        cluster_id += 1
    return labels
