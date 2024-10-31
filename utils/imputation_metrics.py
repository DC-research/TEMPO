import torch
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from torch.nn import MSELoss, L1Loss
import os

def mse_withmask(a,b,mask):
    mse = ((a-b)**2)*mask
    #take mean over all non-batch dimensions
    num_eval = torch.sum(mask,dim=list(range(1,len(a.shape))))
    num_eval[num_eval==0] = 1
    masked_mse = torch.sum(mse,dim=list(range(1,len(a.shape))))/ num_eval
    return torch.mean(masked_mse)

def mae_withmask(a,b,mask):
    mae = torch.abs(a-b)*mask
    #take mean over all non-batch dimensions
    num_eval = torch.sum(mask,dim=list(range(1,len(a.shape))))
    num_eval[num_eval==0] = 1
    masked_mae = torch.sum(mae,dim=list(range(1,len(a.shape))))/ num_eval
    return torch.mean(masked_mae)

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler=0, scaler=1):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler
    # import pdb; pdb.set_trace()
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def main(args):
    df_raw = pd.read_csv(args.data_path)
    df_raw.replace(to_replace=-200, value=np.nan, inplace=True)
    border = [0,int(len(df_raw)*0.8),len(df_raw)]
    cols_data = df_raw.columns[args.time_col:] #change for different data
    df_data = df_raw[cols_data]
    data = df_data.values
    if not args.test:
        data_x = data[border[0]:border[1]]
    else:
        data_x = data[border[1]:border[2]]

    orig_data = []
    seq_len = args.seq_len
    length = len(data_x)-seq_len+1
    for i in range(length):
        orig_data.append(data_x[i:i+seq_len])

    if args.test:
        imputed_path = os.path.join(args.imputed_folder,"test_samples.npy")
        mask_path = os.path.join(args.imputed_folder,"test_masks.npy")
    else:
        imputed_path = os.path.join(args.imputed_folder,"train_samples.npy")
        mask_path = os.path.join(args.imputed_folder,"train_masks.npy")
    synthetic_data = np.load(imputed_path)
    
    scaler = StandardScaler()
    scaler.fit(df_data[border[0]:border[1]].values)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df_data[border[0]:border[1]].values)
    orig_data = np.nan_to_num(orig_data)
    
    target_mask = np.load(mask_path)
    normalized_orig_data = []
    for i in range(len(orig_data)):
        d = orig_data[i]
        d = scaler.transform(d)
        normalized_orig_data.append(d)
    print("MSE: ",MSELoss()(torch.tensor(normalized_orig_data),torch.tensor(synthetic_data)).item())
    print("MAE: ",L1Loss()(torch.tensor(normalized_orig_data),torch.tensor(synthetic_data)).item())
    print("MSE: ",mse_withmask(torch.tensor(normalized_orig_data),torch.tensor(synthetic_data),torch.tensor(target_mask)))
    print("MAE: ",mae_withmask(torch.tensor(normalized_orig_data),torch.tensor(synthetic_data),torch.tensor(target_mask)))

    unormalized_synthetic_data = []
    for i in range(len(synthetic_data)):
        d = synthetic_data[i]
        d = scaler.inverse_transform(d)
        unormalized_synthetic_data.append(d)
    print("MSE: ",MSELoss()(torch.tensor(orig_data),torch.tensor(unormalized_synthetic_data)).item())
    print("MAE: ",L1Loss()(torch.tensor(orig_data),torch.tensor(unormalized_synthetic_data)).item())
    print("MSE: ",mse_withmask(torch.tensor(orig_data),torch.tensor(unormalized_synthetic_data),torch.tensor(target_mask)))
    print("MAE: ",mae_withmask(torch.tensor(orig_data),torch.tensor(unormalized_synthetic_data),torch.tensor(target_mask)))

class ParticipantVisibleError(Exception):
    pass

def WIS_and_coverage(y_true,lower,upper,alpha):

        if np.isnan(lower)  == True:
            raise ParticipantVisibleError("lower interval value contains NaN value(s)")
        if np.isinf(lower)  == True:
            raise ParticipantVisibleError("lower interval value contains inf values(s)")
        if np.isnan(upper)  == True:
            raise ParticipantVisibleError("upper interval value contains NaN value(s)")
        if np.isinf(upper)  == True:
            raise ParticipantVisibleError("upper interval value contains inf values(s)")
        # These should not occur in a competition setting
        if np.isnan(y_true) == True:
            raise ParticipantVisibleError("y_true contains NaN value(s)")
        if np.isinf(y_true) == True:
            raise ParticipantVisibleError("y_true contains inf values(s)")

        # WIS for a single interval
        score = np.abs(upper-lower)
        # print(np.minimum(upper,lower) - y_true)
        if y_true < np.minimum(upper,lower):
            score += ((2/alpha) * (np.minimum(upper,lower) - y_true))
        if y_true > np.maximum(upper,lower):
            score += ((2/alpha) * (y_true - np.maximum(upper,lower)))
        # coverage for one single row
        coverage  = 1
        if (y_true < np.minimum(upper,lower)) or (y_true > np.maximum(upper,lower)):
            coverage = 0
        return score,coverage

v_WIS_and_coverage = np.vectorize(WIS_and_coverage)

def MWIS_score(y_true,lower,upper,alpha):

        # y_true = y_true.astype(float)
        # lower  = lower.astype(float)
        # upper  = upper.astype(float)

        WIS_score,coverage = v_WIS_and_coverage(y_true,lower,upper,alpha)
        MWIS     = np.mean(WIS_score)
        coverage = coverage.sum()/coverage.shape[0]

        MWIS      = float(MWIS)
        coverage  = float(coverage)

        return MWIS,coverage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,default="datasets/AirQualityUCI.csv")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--time_col", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imputed_folder", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)