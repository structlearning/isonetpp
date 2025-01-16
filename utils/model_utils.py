import torch
from GMN.segment import unsorted_segment_sum
import torch.nn.functional as F

def get_graph_features(graphs):
    return graphs.node_features, graphs.edge_features, graphs.from_idx, graphs.to_idx, graphs.graph_idx    

def sample_gumbel(shape, device, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape, device=device, dtype=torch.float)
    return -torch.log(eps - torch.log(U + eps))

def sinkhorn_iters(log_alpha, device, temperature=0.1, noise_factor=1.0, num_iters=20):
    batch_size, num_objs, _ = log_alpha.shape
    noise = sample_gumbel([batch_size, num_objs, num_objs], device) * noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha, temperature)
    for _ in range(num_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, num_objs, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, num_objs)
    return torch.exp(log_alpha)

def attention(log_alpha, temperature=0.1):
    log_alpha = torch.div(log_alpha, temperature)
    attention_q_to_c = log_alpha.softmax(dim = -1)
    attention_c_to_q = log_alpha.softmax(dim = -2).transpose(-1, -2)
    return attention_q_to_c, attention_c_to_q

def pad_log_alpha_by_one(log_alpha):
    batch_size, num_objs, _ = log_alpha.shape
    modified_log_alpha = torch.full(size=(batch_size, num_objs + 1, num_objs + 1), fill_value=0., device=log_alpha.device)
    modified_log_alpha[:, :-1, :-1] = log_alpha
    return modified_log_alpha

def masked_attention(log_alpha, query_sizes, corpus_sizes, temperature=0.1):
    log_alpha = torch.div(log_alpha, temperature)

    batch_size, num_objs, _ = log_alpha.shape
    modified_log_alpha = pad_log_alpha_by_one(log_alpha)

    query_mask = torch.arange(num_objs + 1, device=log_alpha.device).unsqueeze(0) < query_sizes.unsqueeze(1)
    corpus_mask = torch.arange(num_objs + 1, device=log_alpha.device).unsqueeze(0) < corpus_sizes.unsqueeze(1)
    non_pad_mask = torch.bitwise_and(query_mask.unsqueeze(-1), corpus_mask.unsqueeze(-2))
    pad_mask = torch.bitwise_not(torch.bitwise_or(query_mask.unsqueeze(-1), corpus_mask.unsqueeze(-2)))
    overall_mask = torch.bitwise_or(non_pad_mask, pad_mask)
    modified_log_alpha.masked_fill_(torch.bitwise_not(overall_mask), -torch.inf)

    attention_q_to_c = modified_log_alpha.softmax(dim = -1)[:, :-1, :-1]
    attention_c_to_q = modified_log_alpha.softmax(dim = -2)[:, :-1, :-1].transpose(-1, -2)
    return attention_q_to_c, attention_c_to_q

def graph_size_to_mask_map(max_set_size, lateral_dim, device=None):
    return [torch.cat((
        torch.tensor([1], device=device, dtype=torch.float).repeat(x, 1).repeat(1, lateral_dim),
        torch.tensor([0], device=device, dtype=torch.float).repeat(max_set_size - x, 1).repeat(1, lateral_dim)
    )) for x in range(0, max_set_size + 1)]

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_padded_indices(paired_sizes, max_set_size, device):
    num_pairs = len(paired_sizes)
    max_set_size_arange = torch.arange(max_set_size, dtype=torch.long, device=device).reshape(1, -1).repeat(num_pairs * 2, 1)
    flattened_sizes = torch.tensor(flatten_list_of_lists(paired_sizes), device=device)
    presence_mask = max_set_size_arange < flattened_sizes.unsqueeze(1)

    cumulative_set_sizes = torch.cumsum(torch.tensor(
        max_set_size, dtype=torch.long, device=device
    ).repeat(len(flattened_sizes)), dim=0)
    max_set_size_arange[1:, :] += cumulative_set_sizes[:-1].unsqueeze(1)
    return max_set_size_arange[presence_mask]

def split_to_query_and_corpus(features, graph_sizes):
    # [(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
    flattened_graph_sizes  = flatten_list_of_lists(graph_sizes)
    features_split = torch.split(features, flattened_graph_sizes, dim=0)
    features_query = features_split[0::2]
    features_corpus = features_split[1::2]
    return features_query, features_corpus

def split_and_stack(features, graph_sizes, max_set_size):
    features_query, features_corpus = split_to_query_and_corpus(features, graph_sizes)
    
    stack_features = lambda features_array: torch.stack([
        F.pad(features, pad=(0, 0, 0, max_set_size - features.shape[0])) for features in features_array
    ])
    return stack_features(features_query), stack_features(features_corpus)

def feature_alignment_score(query_features, corpus_features, transport_plan):
    return - torch.maximum(
        query_features - transport_plan @ corpus_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(1, 2))

def reversed_feature_alignment_score(query_features, corpus_features, transport_plan):
    return - torch.maximum(
        transport_plan.transpose(-1, -2) @ query_features - corpus_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(1, 2))

def get_paired_edge_counts(from_idx, to_idx, graph_idx, num_graphs):
    edges_per_src_node = unsorted_segment_sum(torch.ones_like(from_idx, dtype=torch.float), from_idx, len(graph_idx))
    edges_per_graph_from = unsorted_segment_sum(edges_per_src_node, graph_idx, num_graphs)

    edges_per_dest_node = unsorted_segment_sum(torch.ones_like(to_idx, dtype=torch.float), to_idx, len(graph_idx))
    edges_per_graph_to = unsorted_segment_sum(edges_per_dest_node, graph_idx, num_graphs)

    assert (edges_per_graph_from == edges_per_graph_to).all()

    return edges_per_graph_to.reshape(-1, 2).int().tolist()

def propagation_messages(propagation_layer, node_features, edge_features, from_idx, to_idx):
    edge_src_features = node_features[from_idx]
    edge_dest_features = node_features[to_idx]

    forward_edge_msg = propagation_layer._message_net(torch.cat([
        edge_src_features, edge_dest_features, edge_features
    ], dim=-1))
    backward_edge_msg = propagation_layer._reverse_message_net(torch.cat([
        edge_dest_features, edge_src_features, edge_features
    ], dim=-1))
    return forward_edge_msg + backward_edge_msg

def kronecker_product_on_nodes(node_transport_plan, from_idx, to_idx, paired_edge_counts, graph_sizes, max_edge_set_size):
    flattened_edge_counts = flatten_list_of_lists(paired_edge_counts)
    segregated_edge_vertices = torch.split(
        torch.cat([from_idx.unsqueeze(-1), to_idx.unsqueeze(-1)], dim=-1),
        flattened_edge_counts, dim=0
    )
    batched_edge_vertices_shifted = torch.cat([
        F.pad(z_idxs, pad=(0, 0, 0, max_edge_set_size - len(z_idxs)), value=-1).unsqueeze(0)
    for z_idxs in segregated_edge_vertices]).long()

    flattened_graph_sizes = flatten_list_of_lists(graph_sizes)
    node_count_prefix_sum = torch.zeros(len(flattened_graph_sizes), device=to_idx.device, dtype=torch.long)
    node_count_prefix_sum[1:] = torch.cumsum(torch.tensor(flattened_graph_sizes, device=to_idx.device), dim=0)[:-1]
    
    batched_edge_vertices = batched_edge_vertices_shifted - node_count_prefix_sum.view(-1, 1, 1)
    batched_edge_vertices[batched_edge_vertices < 0] = -1

    query_from_vertices = batched_edge_vertices[::2, :, 0].unsqueeze(-1)
    query_to_vertices = batched_edge_vertices[::2, :, 1].unsqueeze(-1)
    corpus_from_vertices = batched_edge_vertices[1::2, :, 0].unsqueeze(1)
    corpus_to_vertices = batched_edge_vertices[1::2, :, 1].unsqueeze(1)
    batch_arange = torch.arange(len(graph_sizes), device=to_idx.device, dtype=torch.long).view(-1, 1, 1)
    edge_presence_mask = (query_from_vertices >= 0) * (corpus_from_vertices >= 0)

    straight_mapped_scores = torch.mul(
        node_transport_plan[batch_arange, query_from_vertices, corpus_from_vertices],
        node_transport_plan[batch_arange, query_to_vertices, corpus_to_vertices]
    ) * edge_presence_mask
    cross_mapped_scores = torch.mul(
        node_transport_plan[batch_arange, query_from_vertices, corpus_to_vertices],
        node_transport_plan[batch_arange, query_to_vertices, corpus_from_vertices]
    ) * edge_presence_mask
    return straight_mapped_scores, cross_mapped_scores

def get_interaction_feature_store(transport_plan, query_features, corpus_features, reverse_transport_plan=None):
    if reverse_transport_plan is None:
        reverse_transport_plan = transport_plan.transpose(-1, -2)
    batch_size, set_size, feature_dim = query_features.shape
    assert query_features.shape == corpus_features.shape, "Query and corpus features have different feature dimensions"

    query_from_corpus = torch.bmm(transport_plan, corpus_features)
    corpus_from_query = torch.bmm(reverse_transport_plan, query_features)
    interleaved_features = torch.cat([
        query_from_corpus.unsqueeze(1),
        corpus_from_query.unsqueeze(1)
    ], dim=1).reshape(2*batch_size*set_size, feature_dim)

    return interleaved_features

def features_to_transport_plan(
    query_features, corpus_features, preprocessor, alignment_function, what_for,
    # arguments below can be set exactly once during every forward pass with functools.partial
    graph_size_to_mask_map, query_sizes, corpus_sizes
):
    transformed_features_query = preprocessor(query_features)
    transformed_features_corpus = preprocessor(corpus_features)

    def mask_graphs(features, graph_sizes):
        mask = torch.stack([graph_size_to_mask_map[what_for][i] for i in graph_sizes])
        return mask * features
    masked_features_query = mask_graphs(transformed_features_query, query_sizes)
    masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

    log_alpha = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
    transport_plan = alignment_function(log_alpha=log_alpha, query_sizes=query_sizes, corpus_sizes=corpus_sizes)
    return transport_plan

def consistency_scoring(
    graphs, graph_sizes, max_edge_set_size, node_transport_plan,
    edge_features, device, sinkhorn_config
):
    _, _, from_idx, to_idx, graph_idx = get_graph_features(graphs)

    paired_edge_counts = get_paired_edge_counts(
        from_idx, to_idx, graph_idx, 2*len(graph_sizes)
    )

    stacked_features_query, stacked_features_corpus = split_and_stack(
        edge_features, paired_edge_counts, max_edge_set_size
    )

    straight_mapped_scores, cross_mapped_scores = kronecker_product_on_nodes(
        node_transport_plan=node_transport_plan, from_idx=from_idx,
        to_idx=to_idx, paired_edge_counts=paired_edge_counts,
        graph_sizes=graph_sizes, max_edge_set_size=max_edge_set_size
    )

    edge_transport_plan = sinkhorn_iters(
        log_alpha=straight_mapped_scores+cross_mapped_scores,
        device=device, **sinkhorn_config
    )

    return feature_alignment_score(
        query_features=stacked_features_query,
        corpus_features=stacked_features_corpus,
        transport_plan=edge_transport_plan
    )