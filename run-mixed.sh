source ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com
torchrun --nnodes 1 --nproc-per-node 2 train_mixed.py
