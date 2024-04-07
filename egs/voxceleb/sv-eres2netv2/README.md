## ERes2Net

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpus(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF(p-target=0.01)

### Largin-margin-finetune
- Feature: 80-dim fbank, mean normalization
- Training: lr [2e-5, 1e-4], batch_size 128, 4 gpus(Tesla V100), additive angular margin=0.5

### Voxceleb results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| ERes2NetV2 | 17.8M | 0.68 |  0.065 |
| ERes2NetV2-lm | 17.8M | 0.61  |  0.054 |

### Pretrained model
Waiting...
