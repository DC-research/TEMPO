#!/bin/bash
#SBATCH --job-name=72m2m_np          # Job name
#SBATCH --output=output.6domain_96m2m_no_pool_%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=20G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs
# 
# export CUDA_VISIBLE_DEVICES=2

seq_len=336
model=TEMPO #TEMPO #PatchTST 
electri_multiplier=1
traffic_multiplier=1


for percent in 100 
do
for pred_len in  336 720 
do
for tmax in 20
do
for lr in 0.001 
do
for gpt_layer in 3 
do
for equal in 1 
do
for prompt in 1 
do
mkdir -p logs/$model
mkdir logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/
mkdir logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/ettm2_pmt1_no_pool_$model'_'$gpt_layer
echo logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/ettm2_pmt1_no_pool_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log



python main_multi_6domain_release.py \
    --datasets ETTm1,ETTh1,ETTm2,electricity,ETTh2,weather \
    --target_data traffic \
    --config_path ./configs/multiple_datasets.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./lora_revin_6domain_checkpoints'_'$prompt/ \
    --model_id traffic_TEMPO'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --prompt $prompt\
    --batch_size 256 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 1 \
    --patch_size 16 \
    --stride 8 \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --tmax $tmax \
    --cos 1 \
    --is_gpt 1 #>> logs/$model/loar_revin_$percent'_'percent'_'$prompt'_'prompt'_'equal'_'$equal/ettm2_pmt1_no_pool_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log


done
done
done
done
done
done
done
