# RDINO

## Training config
- Feature: 80-dim fbank
- Training: batch_size 52 * 4, 4 gpu(Tesla V100)
- Metrics: EER(%), MinDCF(p-target=0.05)

## Voxceleb Results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | EER(%) | MinDCF |
|:-----:|:------:|:------:|
| Model performance in the paper |
| RDINO |  3.29  |  0.247 |
| Model perforance reprodunced with this code |
| RDINO |  3.05  |  0.220 |

Note: The original checkpoint is uploaded to ModelScope. The batchsize would affect the learning rate and the number of iterations. It could get the same or similar results if the parameters are unchanged.

## pretrained model
Voxceleb: [ModelScope](https://www.modelscope.cn/home)：[damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k](https://www.modelscope.cn/models/damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k/summary)