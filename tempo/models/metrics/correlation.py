"""Mean Correlation Coefficient from Hyvarinen & Morioka
   Taken from https://github.com/bethgelab/slow_disentanglement/blob/master/mcc_metric/metric.py
"""
import numpy as np
import scipy as sp
import scipy.stats as stats
from .munkres import Munkres

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort

def compute_mcc(mus_train, ys_train, correlation_fn):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  result = np.zeros(mus_train.shape)
  result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
  for i in range(len(mus_train) - len(ys_train)):
    result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
  corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
  mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
  return mcc

#%% MMD
'''
Maximum Mean Discrepancy: https://blog.csdn.net/sinat_34173979/article/details/105876584
'''
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 
  
def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                             	kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size] # Source<->Source
    YY = kernels[batch_size:, batch_size:] # Target<->Target
    XY = kernels[:batch_size, batch_size:] # Source<->Target
    YX = kernels[batch_size:, :batch_size] # Target<->Source
    loss = torch.mean(XX + YY - XY -YX) 
    return loss

if __name__ == '__main__':
  N = 100
  D1 = 3
  D2 = 20
  yt = np.random.normal(size = (D1, N))
  y2 = np.stack((yt[1]+1, yt[0], yt[2]),axis=0)
  print(compute_mcc(yt, y2, "Pearson"))