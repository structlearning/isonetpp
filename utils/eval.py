import torch
from sklearn.metrics import average_precision_score

def pairwise_ranking_loss(pred_pos, pred_neg, margin):
    num_pos, dim = pred_pos.shape
    num_neg, _ = pred_neg.shape

    expanded_pred_pos = pred_pos.unsqueeze(1)
    expanded_pred_neg = pred_neg.unsqueeze(0)
    relu = torch.nn.ReLU()
    loss = relu(margin + expanded_pred_neg - expanded_pred_pos)
    mean_loss = torch.mean(loss, dim=(0, 1))

    return mean_loss

def compute_average_precision(model, pos_pairs, neg_pairs, dataset, return_pred_and_labels=False):
    all_pairs = pos_pairs + neg_pairs
    num_pos_pairs, num_neg_pairs = len(pos_pairs), len(neg_pairs)

    predictions = []
    num_batches = dataset.create_custom_batches(all_pairs)
    for batch_idx in range(num_batches):
        batch_graphs, batch_graph_sizes, _, batch_adj_matrices = dataset.fetch_batch_by_id(batch_idx)
        predictions.append(model(batch_graphs, batch_graph_sizes, batch_adj_matrices).data)
    all_predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])

    average_precision = average_precision_score(all_labels, all_predictions.cpu())
    if return_pred_and_labels:
        return average_precision, all_labels, all_predictions
    else:
        return average_precision