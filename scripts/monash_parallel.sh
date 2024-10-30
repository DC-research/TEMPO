#!/bin/bash
#SBATCH --job-name="TEMPO_parallel"
#SBATCH --output="logs/TEMPO_parallel.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdem-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 12:00:00              # To enable the use of up to 8 GPUs          # To enable the use of up to 8 GPUs

export CUDA_VISIBLE_DEVICES=0,1

seq_len=336
model=TEMPO #TEMPO #PatchTST #_multi
electri_multiplier=3 # 3 times more data than the other small samples.
traffic_multiplier=3


for percent in 100 
do
for pred_len in  96 
do
for tmax in 20
do
for lr in 0.001 
do
for gpt_layer in 6 
do
for equal in 1
do
for prompt in 1 
do
mkdir -p logs/$model
mkdir logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/
mkdir logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/Monash_$model'_'$gpt_layer
echo logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/Monash_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log



torchrun --nproc_per_node=2 train_TEMPO_parallel.py \
    --datasets ETTm1,ETTm2 \
    --eval_data ETTm1 \
    --target_data ETTh2 \
    --config_path ./configs/multiple_datasets.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./checkpoints/Monash'_'$prompt/ \
    --model_id Monash_TEMPO'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 0 \
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
    --is_gpt 1 #>> logs/$model/ReVIN_$prompt'_'prompt'_'equal'_'$equal/Monash_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log


done
done
done
done
done
done
done
