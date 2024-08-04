# Synthesizing event semantics for geographical entity representation

This program provides the implementation of our EGERL as described in our paper.

DATA: The python file data-process.py can process the event data and use different strategies.

### Requirements
- Python 3.7
- Pytorch 1.5.0 & CUDA 10.1

### Running commands:

    python -u EGERL.py --dataset ../data --num_iterations 3000 --eval_after 2000 --batch_size 1024 --lr 0.001 --emb_dim 256 --hidden_dim 256 --encoder QGNN --variant D
