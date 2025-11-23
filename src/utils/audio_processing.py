import torch

def rms(x, dim=-1, eps=1e-9):
    return torch.sqrt((x**2).mean(dim=dim) + eps)

def match_rms(target, src):
    rms_t = rms(target)
    rms_s = rms(src)
    return src * (rms_t / (rms_s + 1e-9))