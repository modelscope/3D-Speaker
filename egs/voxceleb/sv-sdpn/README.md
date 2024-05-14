## SDPN

### Training config
- Feature: 80-dim fbank
- Training: batch_size 96 * 4, 4 gpu(Tesla V100)
- Metrics: EER(%), MinDCF(p-target=0.05)

### Voxceleb results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb1-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| RDINO perforance | 57.24M | 1.80  |  0.139 |
