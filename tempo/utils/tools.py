import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from tempo.utils.metrics import metric
from tempo.utils.imputation_metrics import mse_withmask, mae_withmask, calc_quantile_CRPS, calc_quantile_CRPS_sum

import torch.distributions as dist

plt.switch_backend('agg')


from huggingface_hub import hf_hub_download
from io import StringIO

def load_data_from_huggingface(repo_id, filename):
    try:
        # Download the file content directly into memory
        file_content = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=None,  # This ensures the file is not saved locally
            local_dir_use_symlinks=False
        )
        
        # Read the content into a pandas DataFrame
        with open(file_content, 'r') as f:
            pems_bay = pd.read_csv(StringIO(f.read()))
        
        print("Data loaded successfully")
        return pems_bay
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class EarlyStopping_dist:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.dist = dist

    def __call__(self, val_loss, model, path, rank=0):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, rank)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, rank)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, rank=0):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if rank == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
            else:
                torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def vali(model, vali_data, vali_loader, criterion, args, device, itr):
    total_loss = []
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN' or args.model == 'NLinear' or args.model == 'NLinear_multi':
        model.eval()
    elif args.model == 'TEMPO' or args.model == 'TEMPO_t5' or 'multi' in args.model:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.in_layer_trend.eval()
            model.module.in_layer_season.eval()
            model.module.in_layer_noise.eval()
            model.module.out_layer_trend.eval()
            model.module.out_layer_season.eval()
            model.module.out_layer_noise.eval()
        else:
            model.in_layer_trend.eval()
            model.in_layer_season.eval()
            model.in_layer_noise.eval()
            model.out_layer_trend.eval()
            model.out_layer_season.eval()
            model.out_layer_noise.eval()
    elif args.model == 'GPT4TS' or args.model == 'GPT4TS_prompt':
        model.in_layer.eval()
        model.out_layer.eval()
    else:
        model.eval()
        
    with torch.no_grad():
        for i, data in tqdm(enumerate(vali_loader)):

            batch_x, batch_y, batch_x_mark, batch_y_mark = data[0], data[1], data[2], data[3]
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            if args.model == 'GPT4TS_multi' or args.model == 'NLinear_multi' or 'TEMPO' in args.model:
                seq_trend, seq_seasonal, seq_resid = data[4], data[5], data[6]
                seq_trend = seq_trend.float().to(device)
                seq_seasonal = seq_seasonal.float().to(device)
                seq_resid = seq_resid.float().to(device)
                outputs, _ = model(batch_x, itr,  seq_trend, seq_seasonal, seq_resid)
            elif 'former' in args.model or args.model == 'FEDformer' or args.model == 'TimesNet' or args.model == 'LightTS':
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, itr)
            
            # encoder - decoder
            if args.loss_func == 'prob' or args.loss_func == 'negative_binomial':
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                loss = criterion(batch_y, outputs)
            else:
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN' or  args.model == 'NLinear' or  args.model == 'NLinear_multi':
        model.train()
    elif args.model == 'TEMPO' or args.model == 'TEMPO_t5' or 'multi' in args.model:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.in_layer_trend.train()
            model.module.in_layer_season.train()
            model.module.in_layer_noise.train()
            model.module.out_layer_trend.train()
            model.module.out_layer_season.train()
            model.module.out_layer_noise.train()
        else:
            model.in_layer_trend.train()
            model.in_layer_season.train()
            model.in_layer_noise.train()
            model.out_layer_trend.train()
            model.out_layer_season.train()
            model.out_layer_noise.train()
    elif args.model == 'GPT4TS' or args.model == 'GPT4TS_prompt':
        model.in_layer.train()
        model.out_layer.train()
    else:
        model.train()
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def metric_mae_mse(preds, trues):
    mse = ((preds - trues)**2).mean()
    mae = np.abs(preds - trues).mean()
    return mae, mse

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    # mases = []

    # Initialize accumulators for errors
    total_mae = 0
    total_mse = 0
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid = data[0], data[1], data[2], data[3], data[4], data[5], data[6]
            
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            seq_trend = seq_trend.float().to(device)
            seq_seasonal = seq_seasonal.float().to(device)
            seq_resid = seq_resid.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            
            batch_y = batch_y.float()
            if args.model == 'TEMPO' or args.model == 'TEMPO_t5' or 'multi' in args.model:
                outputs, _ = model(batch_x[:, -args.seq_len:, :], itr,  seq_trend[:, -args.seq_len:, :], seq_seasonal[:, -args.seq_len:, :], seq_resid[:, -args.seq_len:, :])
            elif 'former' in args.model or args.model == 'FEDformer' or args.model == 'TimesNet' or args.model == 'LightTS':
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x[:, -args.seq_len:, :], itr)
            
            # outputs = model(batch_x[:, -args.seq_len:, :], itr)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy().astype(np.float16)
            true = batch_y.detach().cpu().numpy().astype(np.float16)
            torch.cuda.empty_cache()

            # Calculate the batch errors
            batch_mae, batch_mse = metric_mae_mse(pred, true)
            
            # Update the total errors
            total_mae += batch_mae * batch_x.size(0)  # Assuming batch_x.size(0) is the batch size
            total_mse += batch_mse * batch_x.size(0)
            n_samples += batch_x.size(0)

            torch.cuda.empty_cache()
            
            # preds.append(pred)
            # trues.append(true)

    # Calculate the average errors
    mae = total_mae / n_samples
    mse = total_mse / n_samples

    print(f'Average MAE: {mae}')
    print(f'Average MSE: {mse}')
    
    return mse, mae

from torch.distributions import NegativeBinomial

def sample_negative_binomial(mu, alpha, num_samples=1):
    """
    Generate samples from a Negative Binomial distribution.
    
    Args:
    mu (torch.Tensor): Mean parameter of the Negative Binomial distribution.
    alpha (torch.Tensor): Dispersion parameter of the Negative Binomial distribution.
    num_samples (int): Number of samples to generate for each mu-alpha pair.
    
    Returns:
    torch.Tensor: Samples from the Negative Binomial distribution.
    """
    # Ensure mu and alpha are positive
    mu = torch.clamp(mu, min=1e-6)
    alpha = torch.clamp(alpha, min=1e-6)

    # Calculate the parameters needed for PyTorch's NegativeBinomial distribution
    r = 1 / alpha  # shape parameter (number of failures)
    p = torch.clamp(1 / (1 + mu * alpha), min=1e-6, max=1-1e-6)  # success probability
    
    # Create the NegativeBinomial distribution
    nb_dist = NegativeBinomial(total_count=r, probs=p)
    
    # Generate samples
    samples = nb_dist.sample((num_samples,))
    
    return samples

def test_probs(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    # mases = []

    # Initialize accumulators for errors
    total_mae = 0
    total_mse = 0
    n_samples = 0

    preds = []
    trues = []
    masks = []
    means = []
    stds = []

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            batch_x, batch_y, batch_x_mark, batch_y_mark = data[0], data[1], data[2], data[3] #, data[4], data[5], data[6]
            batch_x = batch_x.float().to(device) 
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            batch_y = batch_y.float()

            for channel in range(batch_x.shape[-1]):
                if args.model == 'TEMPO' or args.model == 'TEMPO_t5' or 'multi' in args.model:
                    seq_trend = seq_trend.float().to(device)
                    seq_seasonal = seq_seasonal.float().to(device)
                    seq_resid = seq_resid.float().to(device)
                    outputs, _ = model(batch_x[:, -args.seq_len:, channel:channel+1], itr,  seq_trend[:, -args.seq_len:, :], seq_seasonal[:, -args.seq_len:, :], seq_resid[:, -args.seq_len:, :])
                elif 'former' in args.model or args.model == 'FEDformer' or args.model == 'TimesNet' or args.model == 'LightTS':
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x[:, -args.seq_len:,  channel:channel+1], itr)

                if args.loss_func == 'prob':
                    mu, sigma, nu = outputs[0], outputs[1], outputs[2]
                    # Create the Student's t-distribution with the predicted parameters
                    student_t = dist.StudentT(df=nu, loc=mu, scale=sigma)
                    # Generate 30 samples for each prediction
                    num_samples = args.num_samples
                    probabilistic_forecasts = student_t.rsample((num_samples,))
                elif args.loss_func == 'negative_binomial':
                    mu, alpha = outputs[0], outputs[1]
                    probabilistic_forecasts = sample_negative_binomial(mu, alpha, args.num_samples)

                # The shape of probabilistic_forecasts will be (num_samples, batch_size, pred_length)
                preds.append(probabilistic_forecasts.cpu().numpy())
                trues.append(batch_y[:,:, channel:channel+1].cpu().numpy())
                masks.append(batch_x_mark[:,:, channel:channel+1].cpu().numpy())

            torch.cuda.empty_cache()
            
    trues = np.array(trues)
    preds = np.array(preds)
    masks = np.array(masks)
    trues= np.swapaxes(trues.squeeze(), -2, -3)
    unormzalized_gt_data= np.swapaxes(trues.squeeze(), -1, -2)
    masks= np.swapaxes(masks.squeeze(), -2, -3)
    target_mask= np.swapaxes(masks.squeeze(), -1, -2)
    preds= np.transpose(preds.squeeze(), (2, 1, 3, 0))


    low_q = np.quantile(preds,0.05,axis=1)
    high_q = np.quantile(preds,0.95,axis=1)
    mid_q = np.quantile(preds,0.5,axis=1)

    unormalized_synthetic_data = preds

    print('MAE:', mae_withmask(torch.Tensor(unormzalized_gt_data),torch.Tensor(mid_q),torch.Tensor(target_mask)))

    print('MSE:', mse_withmask(torch.Tensor(unormzalized_gt_data),torch.Tensor(mid_q),torch.Tensor(target_mask)))

    # unormzalized_gt_data = np.swapaxes(unormzalized_gt_data, -1, -2)
    # unormalized_synthetic_data = np.swapaxes(unormalized_synthetic_data, -1, -2)
    # target_mask = np.swapaxes(target_mask, -1, -2)

    # low_q = np.quantile(unormalized_synthetic_data,0.05,axis=1)
    # high_q = np.quantile(unormalized_synthetic_data,0.95,axis=1)
    # mid_q = np.quantile(unormalized_synthetic_data,0.5,axis=1)

    
    # unormzalized_gt_data = np.swapaxes(unormzalized_gt_data, -1, -2)
    # unormalized_synthetic_data = np.swapaxes(unormalized_synthetic_data, -1, -2)
    # target_mask = np.swapaxes(target_mask, -1, -2)

    print('CRPS_Sum:', calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data),torch.Tensor(target_mask),mean_scaler=0,scaler=1))

    print('CRPS:', calc_quantile_CRPS(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data),torch.Tensor(target_mask),mean_scaler=0,scaler=1))
   
    
    return preds, trues #mse, mae
