import torch
import numpy as np
from utils.eval import compute_average_precision

def evaluate_model(model, dataset):
    model.eval()

    # Compute global statistics
    pos_pairs, neg_pairs = dataset.pos_pairs, dataset.neg_pairs
    average_precision = compute_average_precision(model, pos_pairs, neg_pairs, dataset)

    # Compute per-query statistics
    num_query_graphs = len(dataset.query_graphs)
    per_query_avg_prec = []
    
    for query_idx in range(num_query_graphs):
        pos_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, pos_pairs))
        neg_pairs_for_query = list(filter(lambda pair: pair[0] == query_idx, neg_pairs))

        if len(pos_pairs_for_query) > 0 and len(neg_pairs_for_query) > 0:
            per_query_avg_prec.append(
                compute_average_precision(model, pos_pairs_for_query, neg_pairs_for_query, dataset)
            )
    mean_average_precision = np.mean(per_query_avg_prec)
    
    return average_precision, mean_average_precision