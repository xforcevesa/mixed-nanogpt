source ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com
torchrun --nnodes 1 --nproc-per-node 1 train_rwkv7.py
