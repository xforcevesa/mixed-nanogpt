########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)

'''
This will load RWKV-7 "Goose" x070.rc4a-2411 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

DTYPE = torch.bfloat16
# DTYPE = torch.half # better

HEAD_SIZE = 64

########################################################################################################
# CUDA Kernel
########################################################################################################

def RWKV7_OP(r, w, k, v, a, b):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)
    return out.view(B, T, C).to(dtype=DTYPE)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ddd = torch.empty(1, 1, args.n_embd)
            self.time_maa_r = nn.Parameter(ddd)
            self.time_maa_w = nn.Parameter(ddd)
            self.time_maa_k = nn.Parameter(ddd)
            self.time_maa_v = nn.Parameter(ddd)
            self.time_maa_a = nn.Parameter(ddd)
            self.time_maa_g = nn.Parameter(ddd)

            self.time_decay = nn.Parameter(torch.empty(1,1,args.dim_att))
            self.time_faaaa = nn.Parameter(torch.empty(self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.empty(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.empty(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, args.dim_att))

            D_AAA_LORA = 64
            self.time_aaa_w1 = nn.Parameter(torch.empty(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(torch.empty(D_AAA_LORA, args.dim_att))

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(torch.empty(args.n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(torch.empty(D_GATE_LORA, args.dim_att))

            if layer_id > 0:
                D_MV_LORA = 32
                self.mv_w1 = nn.Parameter(torch.empty(args.n_embd, D_MV_LORA))
                self.mv_w2 = nn.Parameter(torch.empty(D_MV_LORA, args.dim_att))
                self.time_misc_v = nn.Parameter(torch.empty(1,1,args.n_embd))

            self.time_misc_kkk = nn.Parameter(torch.empty(1,1,args.n_embd))
            self.time_misc_a = nn.Parameter(torch.empty(1,1,args.n_embd))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def forward(self, x, v0):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.time_maa_r
        xw = x + xx * self.time_maa_w
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xa = x + xx * self.time_maa_a
        xg = x + xx * self.time_maa_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v0 = v
        else:
            v = v + (v0 - v) * torch.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        a = torch.sigmoid(self.time_aaaaa + (xa @ self.time_aaa_w1) @ self.time_aaa_w2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.gate_w1) @ self.gate_w2

        kk = k * self.time_misc_kkk
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.time_misc_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        x = self.output(x * g)
        return x, v0
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.time_maa_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.time_maa_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):

        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first
