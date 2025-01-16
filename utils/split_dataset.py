import os
import pickle
import numpy as np
import shutil
from utils.tooling import seed_everything

source_base_path = "large_dataset/"
total_graphs_num = 300
train_len = 180
val_len = 45
test_len = 75
assert total_graphs_num == train_len + val_len + test_len, "graphs don't add up"

mode_ranges = {
    'train': list(range(train_len)),
    'val': list(range(train_len, train_len + val_len)),
    'test': list(range(train_len + val_len, total_graphs_num)),
}

def file_name(dataset_name, mode, base_path=source_base_path):
    return os.path.join(
        base_path,
        "splits",
        mode,
        f"{mode}_{dataset_name}240k_query_subgraphs.pkl"
    )

def load_first_split(dataset_name):
    graphs = []
    relations = []
    for mode in ["train", "val", "test"]:
        graphs.extend(
            pickle.load(open(file_name(dataset_name, mode), 'rb'))
        )
        relations_dict = pickle.load(open(file_name(dataset_name, mode).replace("query_subgraphs", "rel_nx_is_subgraph_iso"), 'rb'))
        relations.extend([relations_dict[idx] for idx in range(len(relations_dict))])

    return graphs, relations

def save_split(dataset_name, base_path, main_split, new_split_indices):
    os.makedirs(os.path.join(base_path, "splits"), exist_ok=True)
    graphs, relations = main_split
    for mode, mode_range in mode_ranges.items():
        if not os.path.exists(os.path.join(base_path, "splits", mode)):
            os.mkdir(os.path.join(base_path, "splits", mode))
        mode_indices = new_split_indices[mode_range]
        graphs_for_mode = [graphs[idx] for idx in range(len(graphs)) if idx in mode_indices]
        relations_for_mode = [relations[idx] for idx in range(len(relations)) if idx in mode_indices]
        relations_dict = {idx: idx_dict for idx, idx_dict in enumerate(relations_for_mode)}

        pickle.dump(graphs_for_mode, open(file_name(dataset_name, mode, base_path), 'wb'))
        pickle.dump(relations_dict, open(file_name(dataset_name, mode, base_path).replace("query_subgraphs", "rel_nx_is_subgraph_iso"), 'wb'))

    shutil.copy(
        os.path.join(source_base_path, "splits", f"{dataset_name}240k_corpus_subgraphs.pkl"),
        os.path.join(base_path, "splits", f"{dataset_name}240k_corpus_subgraphs.pkl"),
    )

def generate_splits(num_tries=100, splits_reqd=2):
    initial_index = np.arange(total_graphs_num)
    indexes_list = [initial_index]
    position_labels_list = [np.array(
        [0 for _ in range(train_len)] + [1 for _ in range(val_len)] + [2 for _ in range(test_len)]
    ), ]

    for _ in range(num_tries-1):
        indexes = initial_index.copy()
        np.random.shuffle(indexes)
        indexes_list.append(indexes)
        position_labels_list.append(position_labels_list[0][indexes])

    overlap_ratio = np.zeros((num_tries, num_tries))
    for i in range(num_tries):
        for j in range(num_tries):
            overlap_ratio[i, j] = (
                (position_labels_list[i] == position_labels_list[j]) * (position_labels_list[i] == 2)
            ).sum()

    # choose the top k best split
    curr_splits = [0]
    for _ in range(splits_reqd):
        overlap_sum = overlap_ratio[curr_splits].sum(axis=0)
        best_split = np.argmin(overlap_sum)
        curr_splits.append(best_split)
        # sneaky heuristic to prevent repetition of split
        overlap_ratio[curr_splits, curr_splits] = test_len

    assert len(set(curr_splits)) == len(curr_splits), "Same split repeated"
    print(overlap_ratio[curr_splits, :][:, curr_splits])

    return [indexes_list[idx] for idx in curr_splits[1:]]

if __name__ == "__main__":
    seed_everything(0)
    new_splits = generate_splits()

    for dataset_name in ["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"]:
        graphs, relations = load_first_split(dataset_name)
        for new_split_idx, new_split in enumerate(new_splits):
            new_base_path = f"large_dataset_split_{new_split_idx+1}"
            save_split(
                dataset_name=dataset_name,
                base_path=new_base_path,
                main_split=(graphs, relations),
                new_split_indices=new_split
            )
