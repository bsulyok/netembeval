import numpy as np
from typing import Any, Callable
import networkx as nx
from pathlib import Path

def geometrical_contgruence_contribution(
    G: nx.Graph,
    coords: dict[Any, list[float]],
    target_vertices: list[Any],
    skipped_vertices: list[Any],
    distance_function: Callable,
    exclude_neighbours: bool
) -> [float, int] :

    assert len(set(skipped_vertices).intersection(set(target_vertices))) == 0

    N = len(G)
    num_targets = len(target_vertices)
    
    spadjm = nx.adjacency_matrix(G)
    spadjm.indices = spadjm.indices.astype(np.int32)
    spadjm.indptr = spadjm.indptr.astype(np.int32)
    
    target_vertex_indices = [idx for idx, n in enumerate(G.nodes) if n in target_vertices]
    skipped_vertex_indices = [idx for idx, n in enumerate(G.nodes) if n in skipped_vertices]
    coords_array = np.array([coords[v] for v in coords])
    
    eval_mask = np.ones((num_targets, N), dtype=bool)
    eval_mask[np.arange(num_targets), target_vertex_indices] = False
    eval_mask[np.tril_indices(num_targets)[1], np.repeat(target_vertex_indices, np.arange(num_targets)+1)] = False
    # eval_mask[:, skipped_vertex_indices] = False
    
    total_path_length = np.zeros((num_targets, N), dtype=float)
    path_multiplicity = np.zeros((num_targets, N), dtype=int)
    path_multiplicity[np.arange(num_targets), target_vertex_indices] = 1
    topological_distance = np.zeros((num_targets, N), dtype=int)
    
    targets = np.arange(num_targets)
    vertices = np.array(target_vertex_indices)
    
    while 0 < len(vertices):
        targets = targets.repeat(np.diff(spadjm[vertices].indptr))
        neighbours = spadjm[vertices].indices
        vertices = vertices.repeat(np.diff(spadjm[vertices].indptr))
    
        unvisited_mask = path_multiplicity[targets, neighbours] == 0
        targets, vertices, neighbours = targets[unvisited_mask], vertices[unvisited_mask], neighbours[unvisited_mask]
    
        t_v_distance = total_path_length[targets, vertices]
        v_n_distance = distance_function(coords_array[vertices], coords_array[neighbours])
        t_n_distance = t_v_distance + v_n_distance * path_multiplicity[targets, vertices]
        
        np.add.at(total_path_length, (targets, neighbours), t_n_distance)
        np.add.at(path_multiplicity, (targets, neighbours), path_multiplicity[targets, vertices])
    
        targets, vertices = np.unique(np.array([targets, neighbours]), axis=1)
    
    endpoint_distance = distance_function(
        x=coords_array[np.repeat(target_vertex_indices, N)],
        y=coords_array[np.tile(np.arange(N), num_targets)]
    )
    endpoint_distance = endpoint_distance.reshape(-1, N)
    
    score = float(np.mean(endpoint_distance[eval_mask] * path_multiplicity[eval_mask] / total_path_length[eval_mask]))

    return score
