## ResNet

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpus(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF(p-target=0.01)

### Voxceleb results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| ResNet34 | 6.34M | 1.04 |  0.095 |


