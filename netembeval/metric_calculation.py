import numpy as np
import networkx as nx
from typing import Callable
from plotly import graph_objects as go
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr


def calculate_metrics_exactly(G: nx.Graph, coords: dict, distance_function: str | Callable) -> dict[str, float]:
    metrics = {}

    N = len(G)
    M = G.size()
    coords_array = np.array([coords[v] for v in coords])
    
    spadjm = nx.adjacency_matrix(G)
    spadjm.indices = spadjm.indices.astype(np.int32)
    spadjm.indptr = spadjm.indptr.astype(np.int32)

    shortest_path_length = shortest_path(spadjm)
    
    r, c = np.triu_indices(N, k=1)
    
    distances = distance_function(coords_array[r], coords_array[c])
    inv_distance = 1 / (1+distances)
    
    is_edge = spadjm[r, c]

    closest = np.argsort(distances)[:M]
    metrics['edge_prediction_precision'] = float(np.mean(is_edge[closest]))
    metrics['roc_auc_score'] = float(roc_auc_score(is_edge, inv_distance))
    metrics['average_precision_score'] = float(average_precision_score(is_edge, inv_distance))

    
    metrics['mapping_accuracy'] = float(spearmanr(distances, shortest_path_length[r, c]).correlation)
    return metrics
    