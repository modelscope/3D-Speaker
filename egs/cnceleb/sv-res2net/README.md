## Res2Net

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpu(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF

### CNCeleb results
- Train set: CNCeleb-dev + CNCeleb2, 2973 speakers
- Test set: CNCeleb-eval

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| ECAPA-TDNN | 20.8 M  | 8.01 | 0.445 |
| Res2Net | 4.03 M  | 7.96 | 0.452 |

### Pretrained model
waiting...
