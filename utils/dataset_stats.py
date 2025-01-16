import os
import numpy as np
import pandas as pd
import pickle
from utils.split_dataset import load_first_split, source_base_path

def get_stats():
    dataset_info = {}

    for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
        corpus_graphs = pickle.load(open(os.path.join(source_base_path, "splits", f"{dataset_name}240k_corpus_subgraphs.pkl"), 'rb'))
        query_graphs, query_relations = load_first_split(dataset_name)
        query_graph_sizes = [g.size() for g in query_graphs]
        corpus_graph_sizes = [g.size() for g in corpus_graphs]
        query_edge_count = [len(g.edges()) for g in query_graphs]
        corpus_edge_count = [len(g.edges()) for g in corpus_graphs]
        pos_total = np.sum([len(dic['pos']) for dic in query_relations])
        neg_total = np.sum([len(dic['neg']) for dic in query_relations])
        pton_ratio = np.mean([len(dic['pos']) / len(dic['neg']) for dic in query_relations])

        dataset_info[dataset_name] = {
            'Mean $|V_q|$': np.round(np.mean(query_graph_sizes), 4),
            'Min $|V_q|$': np.min(query_graph_sizes),
            'Max $|V_q|$': np.max(query_graph_sizes),
            'Mean $|E_q|$': np.round(np.mean(query_edge_count), 4),
            'Mean $|V_c|$': np.round(np.mean(corpus_graph_sizes), 4),
            'Min $|V_c|$': np.min(corpus_graph_sizes),
            'Max $|V_c|$': np.max(corpus_graph_sizes),
            'Mean $|E_c|$': np.round(np.mean(corpus_edge_count), 4),
            '$\operatorname{pairs(1)}$': pos_total,
            '$\operatorname{pairs(0)}$': neg_total,
            '$\\frac{\operatorname{pairs(1)}}{\operatorname{pairs(0)}}$': np.round(pton_ratio, 4),
        }
    
    return dataset_info

if __name__ == "__main__":
    df = pd.DataFrame.from_dict(get_stats()).T
    for column in df.columns:
        if 'Mean' not in column and 'frac' not in column:
            df[column] = df[column].astype(int)
    print(df.to_latex(float_format=lambda x: "%.2f"%x if x > 1 else "%.4f"%x))
