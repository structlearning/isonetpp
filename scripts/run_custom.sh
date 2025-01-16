cd ..

# best seeds for Node-Early
declare -A dataset_seeds=(
   ["aids"]="7762"
   ["mutag"]="7762"
   ["ptc_fm"]="7474"
   ["ptc_fr"]="7762"
   ["ptc_mm"]="7762"
   ["ptc_mr"]="7366"
)

gpus=(0 1 2 3 4 5)
overall_counter=0

for config_file in \
   "configs/edge_early_variants/edge_early_interaction_baseline.yaml" \
   "configs/edge_early_variants/edge_early_interaction.yaml" \
   "configs/rq4_baselines/scoring=agg___tp=attention_pp=identity_when=post.yaml" \
   "configs/rq4_baselines/scoring=agg___tp=attention_pp=identity_when=pre.yaml" \
   "configs/rq4_baselines/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml" \
   "configs/rq4_baselines/scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true.yaml" \
   "configs/rq4_iterative/iterative___scoring=agg___tp=attention_pp=identity_when=post.yaml" \
   "configs/rq4_iterative/iterative___scoring=agg___tp=attention_pp=identity_when=pre.yaml" \
   "configs/rq4_iterative/iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml" \
   "configs/rq4_iterative/iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true.yaml" \
   "configs/gotsim.yaml" \
   "configs/graphsim.yaml" \
   "configs/simgnn.yaml" \
   "configs/neuromatch.yaml" \
   "configs/egsc.yaml" \
   "configs/isonet.yaml" \
   "configs/gmn_embed.yaml" \
   "configs/node_align_node_loss.yaml" \
   "configs/greed.yaml" \
   "configs/h2mn.yaml" \
; do
   # "configs/..." \
   for dataset in "${!dataset_seeds[@]}"; do
      seed="${dataset_seeds[$dataset]}"
      gpu_index=$(( (overall_counter) % ${#gpus[@]} ))

      WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python3 -m subgraph_matching.train \
         --experiment_id rqX_custom_models \
         --experiment_dir experiments/ \
         --model_config_path $config_file \
         --dataset_name $dataset \
         --seed $seed \
         --dataset_size large \
         &

      ((overall_counter++))
      sleep 10s
   done
done
