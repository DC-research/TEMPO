from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test, EarlyStopping_dist
from torch.utils.data import Subset
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.TEMPO import TEMPO
from models.T5 import T54TS
from models.ETSformer import ETSformer


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from numpy.random import choice

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import sys

from omegaconf import OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler


# def setup(rank, world_size):
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# world_size = torch.cuda.device_count()

# # Call this function at the beginning of your script
# setup(rank, world_size)

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def print_dataset_info(data, loader, name="Dataset"):
    print(f"\n=== {name} Information ===")
    print(f"Number of samples: {len(data)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")
    
    
    for attr in ['features', 'targets', 'shape']:
        if hasattr(data, attr):
            print(f"{attr}: {getattr(data, attr)}")
    

    # for batch in loader:
    #     if isinstance(batch, (tuple, list)):
    #         print("\nFirst batch shapes:")
    #         for i, item in enumerate(batch):
    #             print(f"Item {i} shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    #     else:
    #         print(f"\nFirst batch shape: {batch.shape if hasattr(batch, 'shape') else 'N/A'}")
    #     break

def prepare_data_loaders(args, config):
    """
    Prepare train, validation and test data loaders.
    
    Args:
        args: Arguments containing dataset configurations
        config: Configuration dictionary
    
    Returns:
        tuple: (train_data, train_loader, test_data, test_loader, val_data, val_loader)
    """
    
    train_datas = []
    val_datas = []
    min_sample_num = sys.maxsize
    
    # First pass to get validation data and minimum sample number
    for dataset_name in args.datasets.split(','):
        _update_args_from_config(args, config, dataset_name)
        
        train_data, train_loader = data_provider(args, 'train')
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange', 'monash']:
            min_sample_num = min(min_sample_num, len(train_data))
    
    for dataset_name in args.eval_data.split(','):  
        _update_args_from_config(args, config, dataset_name)  
        val_data, val_loader = data_provider(args, 'val') 
        val_datas.append(val_data)

    # Second pass to prepare training data with proper sampling
    for dataset_name in args.datasets.split(','):
        _update_args_from_config(args, config, dataset_name)
        
        train_data, _ = data_provider(args, 'train')
        
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange', 'monash'] and args.equal == 1:
            train_data = Subset(train_data, choice(len(train_data), min_sample_num))
            
        if args.equal == 1:
            if dataset_name == 'electricity' and args.electri_multiplier > 1:
                train_data = Subset(train_data, choice(len(train_data), 
                                  int(min_sample_num * args.electri_multiplier)))
            elif dataset_name == 'traffic' and args.traffic_multiplier > 1:
                train_data = Subset(train_data, choice(len(train_data),
                                  int(min_sample_num * args.traffic_multiplier)))
                
        train_datas.append(train_data)

    # Combine datasets if multiple exist
    if len(train_datas) > 1:
        train_data = _combine_datasets(train_datas)
        val_data = _combine_datasets(val_datas)
        
    train_sampler = DistributedSampler(train_data, shuffle=True, rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                             num_workers=args.num_workers, sampler=train_sampler)
    
    val_sampler = DistributedSampler(val_data, shuffle=False, rank=dist.get_rank())
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                             num_workers=args.num_workers, sampler=val_sampler)
    
    # Prepare test data
    _update_args_from_config(args, config, args.target_data)
    test_data, test_loader = data_provider(args, 'test')

    print_dataset_info(train_data, train_loader, "Training Dataset")
    print_dataset_info(val_data, val_loader, "Validation Dataset")
    print_dataset_info(test_data, test_loader, "Test Dataset")
    
    return train_data, train_loader, test_data, test_loader, val_data, val_loader, train_sampler, val_sampler

def _update_args_from_config(args, config, dataset_name):
    """Update args with dataset specific configurations"""
    dataset_config = config['datasets'][dataset_name]
    for key in ['data', 'root_path', 'data_path', 'data_name', 'features',
                'freq', 'target', 'embed', 'percent', 'lradj']:
        setattr(args, key, getattr(dataset_config, key))
    
    if args.freq == 0:
        args.freq = 'h'

def _combine_datasets(datasets):
    """Combine multiple datasets into one"""
    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = torch.utils.data.ConcatDataset([combined, dataset])
    return combined




SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}



# torch.cuda.set_device(args.local_rank)
# device  = torch.device("cuda", args.local_rank)
# dist.init_process_group(backend='nccl')
# if dist.is_initialized():
#     rank = dist.get_rank()
# else:
#     rank = 0

def main(args, config):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group(backend='nccl')
    assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    print("rank: ", rank)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")
    mses = []
    maes = []
    args.itr = 1

    print(args)
    for ii in range(args.itr):

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # if args.freq == 0:
        #     args.freq = 'h'

        # Load the data
        train_data, train_loader, test_data, test_loader, vali_data, vali_loader, train_sampler, val_sampler = prepare_data_loaders(args, config)
        print("Data loaded successfully!")
        time_now = time.time()
        train_steps = len(train_loader) #190470 -52696

        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        elif args.model == 'TEMPO':
            model = TEMPO(args, device)
            model.to(device)
        elif args.model == 'T5':
            model = T54TS(args, device)
            model.to(device)
        elif 'ETSformer' in args.model:
            model = ETSformer(args, device)
            model.to(device)
        else:
            model = GPT4TS(args, device)
        # mse, mae = test(model, test_data, test_loader, args, device, ii)
        # model.to(device)
        model = DDP(model.to(device), device_ids=[rank],find_unused_parameters=True)
        params = model.parameters()

        # Assuming you have 4 GPUs
        num_gpus = torch.cuda.device_count()
        # Calculate new effective batch size and learning rate
        effective_batch_size = args.batch_size * num_gpus
        effective_learning_rate = args.learning_rate * num_gpus

        model_optim = torch.optim.Adam(params, lr=effective_learning_rate)
        # model_optim = torch.optim.Adam(params, lr=args.learning_rate)

        # Implement gradual warmup
        warmup_factor = 1.0 / 1000
        warmup_iters = 1000

        def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
            if epoch < 5:  # Warmup for first 5 epochs
                alpha = iteration / (num_iter * 5)
                lr_mult = warmup_factor * (1 - alpha) + alpha
            else:
                lr_mult = 1.0
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = effective_learning_rate * lr_mult
            
        early_stopping = EarlyStopping_dist(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        for epoch in range(args.train_epochs):
            train_sampler.set_epoch(epoch)

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(enumerate(train_loader),total = len(train_loader)):
                
                adjust_learning_rate(model_optim, epoch, i, len(train_loader))
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                seq_trend = seq_trend.float().to(device)
                seq_seasonal = seq_seasonal.float().to(device)
                seq_resid = seq_resid.float().to(device)

                # print(seq_seasonal.shape)
                if args.model == 'TEMPO' or 'multi' in args.model:
                    outputs, loss_local = model(batch_x, ii, seq_trend, seq_seasonal, seq_resid) #+ model(seq_seasonal, ii) + model(seq_resid, ii)
                elif 'former' in args.model:
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, ii)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                loss = criterion(outputs, batch_y) 
                if args.model == 'GPT4TS_multi' or args.model == 'TEMPO_t5':
                    if not args.no_stl_loss:
                        loss += args.stl_weight*loss_local
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
            
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model, path, rank)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        

        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path), strict=False)
        print("------------------------------------")
        mse, mae = test(model, test_data, test_loader, args, device, ii)
        torch.cuda.empty_cache()
        print('test on the ' + str(args.target_data) + ' dataset: mse:' + str(mse) + ' mae:' + str(mae))
        
        mses.append(mse)
        maes.append(mae)


    print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEMPO')

    parser.add_argument('--model_id', type=str, default='weather_GTP4TS_multi-debug')
    parser.add_argument('--checkpoints', type=str, default='/l/users/defu.cao/checkpoints_multi_dataset/')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')


    parser.add_argument('--prompt', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)


    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type3') # for what
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)

    parser.add_argument('--loss_func', type=str, default='mse')
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='GPT4TS_multi')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)

    parser.add_argument('--itr', type=int, default=3)
    parser.add_argument('--cos', type=int, default=0)
    parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
    parser.add_argument('--pool', action='store_true', help='whether use prompt pool')
    parser.add_argument('--no_stl_loss', action='store_true', help='whether use prompt pool')

    parser.add_argument('--stl_weight', type=float, default=0.01)
    parser.add_argument('--config_path', type=str, default='./data_config.yml')
    parser.add_argument('--datasets', type=str, default='exchange')
    parser.add_argument('--target_data', type=str, default='ETTm1')
    #eval_data
    parser.add_argument('--eval_data', type=str, default='exchange')

    parser.add_argument('--use_token', type=int, default=0)
    parser.add_argument('--electri_multiplier', type=int, default=1)
    parser.add_argument('--traffic_multiplier', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')

    

    #args = parser.parse_args([])
    args = parser.parse_args()
    config = get_init_config(args.config_path)

    main(args, config)