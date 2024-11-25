# Mixed NanoGPT with Latest RWKV

## Synopsis

A minimal large language model training structure on Huawei Ascend 970b.

Still on developing ...

## Get Started

Before stepping on the following, please ensure your Ascend 970b is available.

```bash
conda env create -f env.yaml
conda activate PyTorch-2.1.0
python fineweb.py # Download 10B-sized part of fineweb dataset from huggingface
bash run-gpt2.sh # 2 nproc by default, modified according to your demand
bash run-mixed.sh # 2 nproc by default, modified according to your demand
python graph.py # Draw a graph of train & valid loss
```

## Source Code

- `train_gpt2.py`: Ported original NanoGPT training code to Ascend 910b, with better support for DDP training on `hccl` backend and `npu` device and faster training speed as around 309 Ktokens/s.

- `train_mixed.py`: Modified mixed model. The training speed is a little slow.

- `log-old`: The log of train & valid loss of original NanoGPT on 4 nproc DDP, so the values are supposed to be divided by 4.

- `log`: The log of train & valid loss of new mixed model on 2 nproc DDP, so the values are supposed to be divided by 2.
