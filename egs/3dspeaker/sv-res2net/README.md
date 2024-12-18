## Res2Net

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpus(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF(p-target=0.01)

### 3D-Speaker results
- Train set: 3D-Speaker-train
- Test set: 3D-Speaker-test

| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| ECAPA-TDNN | 20.8 M | 8.87% | 12.26% | 14.53% |
| Res2Net | 4.03 M | 8.03% | 9.67% | 14.11% |

### Pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- 3D-Speaker-Dataset: [iic/speech_res2net_sv_zh-cn_3dspeaker_16k](https://modelscope.cn/models/iic/speech_res2net_sv_zh-cn_3dspeaker_16k)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# Res2Net trained on 3D-Speaker-Dataset
model_id=iic/speech_res2net_sv_zh-cn_3dspeaker_16k
# Run inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

