## Res2Net

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 512, 8 gpus(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF(p-target=0.01)

### Voxceleb results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| Res2Net | 4.03M | 1.50  |  0.138 |

