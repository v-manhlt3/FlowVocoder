import numpy as np
import torch
from numba import jit
from torch.autograd import Function
import time

@jit(nopython = True)
def compute_softdtw(D, gamma):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.ones((B, N + 2, M + 2)) * np.inf
  R[:, 0, 0] = 0
  for k in range(B):
    for j in range(1, M + 1):
      for i in range(1, N + 1):
        r0 = -R[k, i - 1, j - 1] / gamma
        r1 = -R[k, i - 1, j] / gamma
        r2 = -R[k, i, j - 1] / gamma
        rmax = max(max(r0, r1), r2)
        rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
        softmin = - gamma * (np.log(rsum) + rmax)
        R[k, i, j] = D[k, i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2))
  E = np.zeros((B, N + 2, M + 2))
  D[:, 1:N + 1, 1:M + 1] = D_
  E[:, -1, -1] = 1
  R[:, : , -1] = -np.inf
  R[:, -1, :] = -np.inf
  R[:, -1, -1] = R[:, -2, -2]
  for k in range(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):
        a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
        b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
        c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  return E[:, 1:N + 1, 1:M + 1]

@jit(nopython = True)
def trace_reverse(trace_matrices):
    b = trace_matrices.shape[0]
    t_1 = trace_matrices.shape[1]
    t_2 = trace_matrices.shape[2]
    alignment_matrices = np.zeros((b, t_1-2))
    matrices_mask = np.zeros((b, t_1, t_2))
    for k in range(trace_matrices.shape[0]):
      trace_matrix = trace_matrices[k]
      max_i = trace_matrix.shape[0]
      max_j = trace_matrix.shape[1]
      # dev = trace_matrix.device
      alignment_vector = np.zeros(max_i)
      i = max_i - 1
      j = max_j - 1
      finished = True
      while finished:
        # print("i: {}----- j: {}".format(i,j))
        diagnol_ele = trace_matrix[i-1, j-1]
        upper_ele = trace_matrix[i-1, j]
        left_ele = trace_matrix[i, j-1]
        # print("in loop soft-dtw")
        # print(diagnol_ele, upper_ele, left_ele)

        if diagnol_ele <= upper_ele and diagnol_ele <= left_ele:
          alignment_vector[i-1] += 1 
          matrices_mask[k][i][j] = 1
          i-=1
          j-=1
        elif left_ele <= diagnol_ele and left_ele <= upper_ele:
          alignment_vector[i] += 1
          matrices_mask[k][i][j] = 1
          j-=1
        elif upper_ele <= diagnol_ele and upper_ele <= left_ele:
          i-=1

        if i == 0 and j == 0:
          finished =False
      
      alignment_vector = alignment_vector[1:-1]
      # alignment_matrices.append(alignment_vector.unsqueeze(0))
      # alignment_matrices.append(np.expand_dims(alignment_vector, 0))
      # print("Alignment vector: ", alignment_vector.shape)
      # print("Alignment matrices: ", alignment_matrices.shape)
      alignment_matrices[k] = np.expand_dims(alignment_vector, 0)

    # print("aligment matices types", type(alignment_matrices))
    # alignment_matrices = tuple(alignment_matrices)
    # alignment_matrices = np.concatenate(alignment_matrices, 0)
    # alignment_matrices = alignment_matrices.unsqueeze(-1)
    alignment_matrices = np.expand_dims(alignment_matrices, -1)
    """
      alignment_matrices: (b, t, d)
      matrices_mask: (b, source_t, target_t)
    """
    # print("alignment matrices shape: ", alignment_matrices.shape)
    # print("matrices mask: ", matrices_mask.shape)
    return alignment_matrices, matrices_mask[:,1:-1,1:-1]


class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item() 
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    
    trace_matrix = R
    # print("R: ", R)
    ctx.save_for_backward(D, R, gamma)
    return R[:, -2, -2], trace_matrix

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output.view(-1, 1, 1).expand_as(E) * E, None

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma=gamma
    self.func_dtw = _SoftDTW.apply

  def calc_distance_matrix(self, x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    dist = torch.pow(x - y, 2).sum(3)
    return dist

  # def forward(self, x, y):
  #   assert len(x.shape) == len(y.shape)
  #   squeeze = False
  #   if len(x.shape) < 3:
  #     x = x.unsqueeze(0)
  #     y = y.unsqueeze(0)
  #     squeeze = True
  #   if self.normalize:
  #     D_xy = self.calc_distance_matrix(x, y)
  #     out_xy,_ = self.func_dtw(D_xy, self.gamma)
  #     D_xx = self.calc_distance_matrix(x, x)
  #     out_xx,_ = self.func_dtw(D_xx, self.gamma)
  #     D_yy = self.calc_distance_matrix(y, y)
  #     out_yy,_ = self.func_dtw(D_yy, self.gamma)
  #     result = out_xy - 1/2 * (out_xx + out_yy) # distance
  #   else:
  #     D_xy = self.calc_distance_matrix(x, y)
  #     out_xy, trace_matrix = self.func_dtw(D_xy, self.gamma)
  #     result = out_xy # discrepancy
  #   return result.squeeze(0) if squeeze else result, trace_matrix

  def forward(self, cost_matrix):

    out_xy, trace_matrix = self.func_dtw(cost_matrix, self.gamma)
    result = out_xy # discrepancy
    return result, trace_matrix

  def trace_reverse(self, trace_matrices):
    alignment_matrices = []
    matrices_mask = torch.zeros_like(trace_matrices).to(trace_matrices.device)

    for k in range(trace_matrices.size(0)):

      trace_matrix = trace_matrices[k]
      max_i = trace_matrix.size(0)
      max_j = trace_matrix.size(1)
      dev = trace_matrix.device
      alignment_vector = torch.zeros(max_i).to(dev)
      i = max_i - 1
      j = max_j - 1
      finished = True
      while finished:
        # print("i: {}----- j: {}".format(i,j))
        diagnol_ele = trace_matrix[i-1, j-1]
        upper_ele = trace_matrix[i-1, j]
        left_ele = trace_matrix[i, j-1]
        # print("in loop soft-dtw")
        # print(diagnol_ele, upper_ele, left_ele)

        if diagnol_ele <= upper_ele and diagnol_ele <= left_ele:
          alignment_vector[i-1] += 1 
          matrices_mask[k][i][j] = 1
          i-=1
          j-=1
        elif left_ele <= diagnol_ele and left_ele <= upper_ele:
          alignment_vector[i] += 1
          matrices_mask[k][i][j] = 1
          j-=1
        elif upper_ele <= diagnol_ele and upper_ele <= left_ele:
          i-=1

        if i == 0 and j == 0:
          finished =False
      alignment_vector = alignment_vector[1:-1]
      alignment_matrices.append(alignment_vector.unsqueeze(0))

    alignment_matrices = torch.cat(alignment_matrices, 0)
    alignment_matrices = alignment_matrices.unsqueeze(-1)
    """
      alignment_matrices: (b, t, d)
      matrices_mask: (b, source_t, target_t)
    """
    return alignment_matrices, matrices_mask[:,1:-1,1:-1] 

####################################################################################################
# soft_dtw = SoftDTW().cuda()
# source = torch.randn(32, 140, 80).cuda()
# target = torch.randn(32, 800, 80).cuda()
# # source = torch.tensor([[1.,1.],[1.,2.],[2.,2.]])
# # source = source.unsqueeze(0)
# # target = torch.tensor([[1.1,1.1],[1.1,1.1],[1.1,2.1],[1.1,1.1],[2.1,2.1]])
# # target = target.unsqueeze(0)

# # # print(target.shape, source.shape)
# t_begin1 = time.time()
# cost_matrix = soft_dtw.calc_distance_matrix(source, target)
# t_end1 = time.time()

# result, trace_matrices = soft_dtw(cost_matrix)
# # print(result)
# # # print(trace_matrix[:,-2, -2])
# trace_matrices = trace_matrices.detach().cpu().numpy()
# t_begin2 = time.time()
# aligment_matrix, matrices_mask = trace_reverse(trace_matrices)
# t_end2 = time.time()

# print("time to compute costmatrix:  ", (t_end1 - t_begin1))
# print("time to compute alignment matrix:  ", (t_end2 - t_begin2))

# matrices_mask = torch.transpose(matrices_mask, -1, -2)
# # source = torch.transpose(source, -1, -2)
# print(source.shape, matrices_mask.shape)
# final_alignment_vector = torch.matmul(matrices_mask, source)
# # aa = torch.sum(aligment_vector)
# # print(trace_matrix.shape)
# # print(trace_matrix[:,-2, -2])

# print(aligment_matrix.shape)
# print(final_alignment_vector.view(1, 5, 2))
# print(matrices_mask)

