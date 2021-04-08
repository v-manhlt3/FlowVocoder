import torch
import math
from math import log, pi

class WaveFlowLossDataParallel(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveFlowLossDataParallel, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        out, logdet,_ = model_output
        logdet = logdet.sum().float()
        B, h, C, T = out.size()

        loss = (0.5) * (log(2.0 * pi) + 2 * log(self.sigma) + out.pow(2) / (self.sigma*self.sigma)).sum() - logdet
        return loss / (B*C*T)

class DistillationLoss(torch.nn.Module):
    def __init__(self, compress_factor=2):
        super(DistillationLoss, self).__init__()
        self.compress = compress_factor

    def forward(self, t_logdet, s_logdet):
        # compress_t_logdet = torch.zeros_like(s_logdet).to(s_logdet.device)
        # s_logdet = [s_logdet[i].sum() for i in range(len(s_logdet))]
        loss = 0
        for i in range(0, len(t_logdet), int(self.compress)):
            # logdet = 0
            for j in range(i, int(self.compress)):
                loss += torch.abs(t_logdet[j] - s_logdet[i])
            # compress_t_logdet[i] = logdet

        # loss = torch.abs(compress_t_logdet - s_logdet)
        return loss
        # t_logdet_ = [logdet[i] for i in range(len(t_logdet), self.c)]