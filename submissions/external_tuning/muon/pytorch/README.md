
### Momentum
Which momentum implementation should we use?
(A) w/ dampening (1-beta):
    - KellerJordan/Muon: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L35
    - KellerJordan/modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt/blob/1b51e26d304f647c7c12201b3f1513ee5a429ec4/train_gpt.py#L197
(B) w/out dampening:
    - KellerJordan/cifar10: https://github.com/KellerJordan/cifar10-airbench/blob/0e6f9614572d7e8e3c259905aebc7196f91d5d79/research/clean_muon.py#L91
    - original KellerJordan/modded-nanogpt: https://github.com/microsoft/dion/blob/0360f9b0369603ecfa19de5128f56c983f1ac7d9/dion/muon_reference.py#L323
    - MoonShootAI: https://arxiv.org/pdf/2502.16982
    - Dion: https://github.com/microsoft/dion/tree/main
We allow both by specifying two momentum hyperparameters: `muon_beta` and `muon_dampening`.

### Correctly assign AlgoPerf parameters to Muon/AdamW
- criteo1tb: `embedding_chunk_{i}` -> dense & sparse features -> AdamW

### TODO
- check for dropout correctness
- how should we handle weight decay?
- seprate learning rate and weight decay for AdamW and Muon?
- perhaps add signum backup(Scion) and Lion backup
