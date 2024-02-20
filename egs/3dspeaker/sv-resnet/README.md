## ResNet

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpus(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF(p-target=0.01)

### 3D-Speaker Results
- Train set: 3D-Speaker-train
- Test set: 3D-Speaker-test

| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| ECAPA-TDNN | 20.8 M | 8.87% | 12.26% | 14.53% |
| ResNet34 | 6.34 M | 7.29% | 8.98% | 12.81% |

### pretrained model
waiting...