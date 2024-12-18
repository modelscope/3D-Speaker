## ERes2NetV2

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpus(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF(p-target=0.01)

### Largin-margin-finetune
- Feature: 80-dim fbank, mean normalization
- Training: lr [2e-5, 1e-4], batch_size 128, 4 gpus(Tesla V100), additive angular margin=0.5

### Voxceleb results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb1-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| ERes2NetV2 | 17.8M | 0.68 |  0.065 |
| ERes2NetV2-lm | 17.8M | 0.61  |  0.054 |

### Pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- 200k labeled speakers: [iic/speech_eres2netv2_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# ERes2NetV2 trained on 200k labeled speakers
model_id=iic/speech_eres2netv2_sv_zh-cn_16k-common
# Run inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

### Citations
If you are using ERes2NetV2 model in your research, please cite: 
```BibTeX
@article{chen2024eres2netv2,
  title={ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational Efficiency},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and and others},
  booktitle={INTERSPEECH},
  year={2024}
}
```
