source ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com
torchrun --nnodes 1 --nproc-per-node 1 test_efficiency_fp32_moba.py
