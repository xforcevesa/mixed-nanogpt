import os
import math
import time
import types
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from rwkv_v7_fp32 import Block as RWKVBlock
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_cluster = config.n_cluster
        self.group_size = config.group_size
        # 
        self.soft_assign = nn.Linear(config.n_embd, config.n_cluster, bias=False)

    def apply_rotary_position_embeddings(self, sinusoidal_pos, x):
        # Split the sinusoidal_pos into sin and cos parts
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # Apply the rotary embeddings
        x_rot = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1)
        x_rot = torch.reshape(x_rot, x.shape[:-1] + (x.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
        x_rot = torch.reshape(x_rot, x.shape)
        return x_rot

    def get_sinusoidal_embeddings(self, n_positions, dim):
        """Generate sinusoidal positional embeddings."""
        position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        sinusoidal_emb = torch.zeros((n_positions, dim))
        sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
        return sinusoidal_emb

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        return self.short_forward(x) if T <= self.group_size else self.long_forward(x)

    def short_forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary position embeddings
        sinusoidal_pos = self.get_sinusoidal_embeddings(T, self.n_embd // self.n_head).to(x.device)
        q_rot = self.apply_rotary_position_embeddings(sinusoidal_pos, q)
        k_rot = self.apply_rotary_position_embeddings(sinusoidal_pos, k)

        y = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

    def long_forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        device = x.device
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # step 1: dynamic assignment and clustering
        cluster_assign = self.soft_assign(x).permute(0, 2, 1) # (B, n_cluster, T)
        # Calculate the number of tokens per cluster
        num_tokens_per_cluster = T // self.n_cluster
        assert T % self.n_cluster == 0, f"sequence with {T} tokens cannot be divided into {self.n_cluster} clusters"
        # Create a mask tensor filled with zeros, shape (B, n_cluster, T)
        attn_mask = torch.zeros((B, self.n_cluster, T), dtype=torch.bool, device=device)
        # Generate token indices
        token_indices = torch.arange(T, device=device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T)
        # Compute the start and end indices for each cluster
        start_indices = torch.arange(0, T, num_tokens_per_cluster, device=device).unsqueeze(1)  # Shape: (n_cluster, 1)
        end_indices = start_indices + num_tokens_per_cluster  # Shape: (n_cluster, 1)
        # Use broadcasting to generate the mask
        attn_mask[:, :, :] = (token_indices >= start_indices) & (token_indices < end_indices)
        # Apply the mask to the cluster assignment
        cluster_assign.masked_fill_(attn_mask.logical_not(), float("-inf"))
        # softmax over assignment:
        cluster_assign = F.softmax(cluster_assign, dim=-1) # (B, n_cluster, T)
        # apply the assignment to the keys and values, get dynamic centers of k and v
        ck = cluster_assign @ k # (B, n_cluster, T) @ (B, T, C) -> (B, n_cluster, C)
        cv = cluster_assign @ v # (B, n_cluster, T) @ (B, T, C) -> (B, n_cluster, C)

        # step 2: group seqence into groups
        n_group = T // self.group_size
        assert T % self.group_size == 0, "T must be divisible by group_size in training mode"
        # reshape q, k, v to (B, n_group, group_size, C)
        q = q.view(B, n_group, self.group_size, C) # (B, n_group, group_size, C)
        k = k.view(B, n_group, self.group_size, C)
        v = v.view(B, n_group, self.group_size, C)
        # step 3: apply causal attention to each group
        num_clusters_per_group = self.group_size // num_tokens_per_cluster # how many clusters in a group
        y = torch.zeros_like(q) # (B, n_group, group_size, C)
        for i in range(n_group):
            qi = q[:, i, :, :].view(B, self.group_size, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, gs, hs)
            num_cluster_tokens = num_clusters_per_group * i # [0, 1, 2, ..., n_group-1] * num_clusters_per_group
            if num_cluster_tokens == 0:
                ki = k[:, i, :, :].view(B, self.group_size, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, gs, hs)
                vi = v[:, i, :, :].view(B, self.group_size, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, gs, hs)
            else:
                ck_i = ck[:, :num_cluster_tokens, :]
                cv_i = cv[:, :num_cluster_tokens, :]
                ki = torch.cat([ck_i, k[:, i, :, :]], dim=1).view(B, self.group_size+num_cluster_tokens, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, gs+ct, hs)
                vi = torch.cat([cv_i, v[:, i, :, :]], dim=1).view(B, self.group_size+num_cluster_tokens, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, gs+ct, hs)
            # apply rotary position embeddings
            q_sinusoidal_pos = self.get_sinusoidal_embeddings(qi.size(2), C // self.n_head).to(device)
            k_sinusoidal_pos = self.get_sinusoidal_embeddings(ki.size(2), C // self.n_head).to(device)
            qi_rot = self.apply_rotary_position_embeddings(q_sinusoidal_pos, qi)
            ki_rot = self.apply_rotary_position_embeddings(k_sinusoidal_pos, ki)
            # create causal mask
            causal_group_mask = torch.ones(self.group_size, self.group_size, dtype=torch.bool, device=device).tril(diagonal=0) # (gs, gs)
            if num_cluster_tokens == 0:
                causal_mask = causal_group_mask
            else:
                causal_cluster_mask = torch.ones(self.group_size, num_cluster_tokens, dtype=torch.bool, device=device) # (gs, ct)
                causal_mask = torch.cat([causal_cluster_mask, causal_group_mask], dim=1) # (group_size, group_size+num_cluster_tokens)
            # calculate attention
            yi = F.scaled_dot_product_attention(qi_rot, ki_rot, vi, attn_mask=causal_mask) # flash attention
            yi = yi.transpose(1, 2).contiguous().view(B, self.group_size, C) # re-assemble all head outputs side by side
            y[:, i, :, :] = yi
        # step 4: recompose groups into sequence
        y = y.view(B, T, C) # (B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    n_cluster: int = 256 # number of clusters
    group_size: int = 256 # group size
    chunk_len: int = 16 # don't change
    interval: int = 4 # 1 Transformer block every 4 blocks

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        args = types.SimpleNamespace()

        args.head_size_a = 64 # don't change
        args.head_size_divisor = 8 # don't change
        args.n_embd = config.n_embd
        args.dim_att = config.n_embd
        args.dim_ffn = config.n_embd * 4
        args.n_layer = config.n_layer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) if i % config.interval == (config.interval-1) else RWKVBlock(args, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        if T % self.config.chunk_len != 0:
            # left padding by add eos token at the beginning, 50256 is the eos token of gpt2 tokenizer
            num_tokens_to_pad = self.config.chunk_len - T % self.config.chunk_len
            eos_idx = torch.full((B, num_tokens_to_pad), 50256, dtype=torch.long, device=idx.device)
            idx = torch.cat((eos_idx, idx), dim=-1)
        # forward the token embeddings
        x = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # forward the blocks of the transformer
        v_first = torch.empty_like(x)
        for block in self.transformer.h:
            if isinstance(block, RWKVBlock):
                x, v_first = block(x, v_first)
            else:
                x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # if left padding was done, remove the leftmost elements
        if T % self.config.chunk_len != 0:
            logits = logits[:, num_tokens_to_pad:]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"total number of parameter tensors: {len(param_dict)}, with {num_decay_params + num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    # assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cuda"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 131072*3# ~0.39M, in number of tokens; old is 524288 = 2**19
B = 4*2 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
if master_process:
    print(model.transformer)
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 2e-4
min_lr = max_lr * 0.1
warmup_steps = 715
#max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
max_steps = 25431 # 25,431 steps is ~1 epoch, if data is 10B tokens and batch size 0.39M tokens
if master_process:
    print(f"max_lr: {max_lr}, min_lr: {min_lr}, warmup_steps: {warmup_steps}, max_steps: {max_steps}")
    print(f"total tokens will be trained: {total_batch_size * max_steps / 1e9:.2f}B")
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log/lmh_L12D768_CTX1024_C256G256_INTV4_ROPE"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float32):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
        if master_process:
            val_loss_accum = val_loss_accum.item() / ddp_world_size
            print(f"validation loss: {val_loss_accum:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} {step*total_batch_size} val {val_loss_accum:.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float32):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} {step*total_batch_size} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float32):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.float32):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        loss_accum = loss_accum.item() / ddp_world_size
        print(f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} {step*total_batch_size} train {loss_accum:.6f}\n")

if ddp:
    destroy_process_group()
