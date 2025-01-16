import os
import time
import torch
import wandb
from utils.parser import Parser
import utils.early_stopping as es
from utils.experiment import Experiment
from utils.eval import pairwise_ranking_loss
from subgraph_matching.test import evaluate_model
from subgraph_matching.dataset import get_datasets
from utils.tooling import seed_everything, read_config
from subgraph_matching.model_handler import get_model, get_data_type_for_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def train_model(
    model: torch.nn.Module, datasets, experiment: Experiment, early_stopping: es.EarlyStopping,
    max_epochs, margin, weight_decay, learning_rate, device
):
    model.to(device)
    
    train_dataset, val_dataset = datasets['train'], datasets['val']
    experiment.save_initial_model_state_dict(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    experiment.log(f"no. of params in model: {num_parameters}")

    for epoch_num in range(max_epochs):
        model.train()
        
        num_batches = train_dataset.create_stratified_batches()
        epoch_loss = 0
        training_start_time = time.time()
        for batch_idx in range(num_batches):
            batch_graphs, batch_graph_sizes, labels, batch_adj_matrices = train_dataset.fetch_batch_by_id(batch_idx)
            optimizer.zero_grad()
            prediction = model(batch_graphs, batch_graph_sizes, batch_adj_matrices)

            predictions_for_positives = prediction[labels > 0.5]
            predictions_for_negatives = prediction[labels < 0.5]
            losses = pairwise_ranking_loss(
                predictions_for_positives.unsqueeze(1),
                predictions_for_negatives.unsqueeze(1),
                margin
            )
            if hasattr(model, 'regularizer'):
                losses += model.regularizer
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        epoch_training_time = time.time() - training_start_time
        experiment.log(f"Run: %d train loss: %f Time: %.2f", epoch_num, epoch_loss, epoch_training_time)

        model.eval()
        validation_start_time = time.time()
        ap_score, map_score = evaluate_model(model, val_dataset)
        epoch_validation_time = time.time() - validation_start_time
        experiment.log(f"Run: %d VAL ap_score: %.6f map_score: %.6f Time: %.2f", epoch_num, ap_score, map_score, epoch_validation_time)

        wandb.log(
            {
            "train_loss": epoch_loss,
            "train_time": epoch_training_time,
            "val_ap_score": ap_score,
            "val_map_score": map_score,
            "val_time": epoch_validation_time,
            },
            step = epoch_num
        )

        es_verdict = early_stopping.check([map_score])
        if es_verdict == es.SAVE:
            experiment.save_best_model_state_dict(model, epoch_num)
        elif es_verdict == es.STOP:
            break
    
    model.load_state_dict(experiment.load_best_model_state_dict())
    return model

if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)

    device = 'cuda' if args.use_cuda else 'cpu'

    model_params, config_dict = read_config(args.model_config_path, with_dict=True)

    experiment_config = parser.get_experiment_config(model_params.name)
    experiment = Experiment(config=experiment_config, device=device)

    wandb_params = parser.get_wandb_config(config_dict, device)
    wandb.init(**wandb_params, resume=False)

    early_stopping_config = parser.get_early_stopping_config()
    early_stopping = es.EarlyStopping(**early_stopping_config)
    
    dataset_config = parser.get_dataset_config()
    data_type = get_data_type_for_model(model_params.name)
    datasets = get_datasets(dataset_config, experiment, data_type)

    model = get_model(
        model_name=model_params.name,
        config=model_params.model_config,
        max_node_set_size=datasets['train'].max_node_set_size,
        max_edge_set_size=datasets['train'].max_edge_set_size,
        device=device
    )

    trained_model = train_model(
        model,
        datasets,
        experiment,
        early_stopping,
        **parser.get_optimization_config(),
        device = device
    )

    test_dataset = get_datasets(dataset_config, experiment, data_type, modes=['test'])['test']
    test_ap_score, test_map_score = evaluate_model(trained_model, test_dataset)
    experiment.log(f"TEST - ap_score: %.6f map_score: %.6f", test_ap_score, test_map_score)
    wandb.log({
        "test_ap_score": test_ap_score,
        "test_map_score": test_map_score,
    })

    wandb.finish()