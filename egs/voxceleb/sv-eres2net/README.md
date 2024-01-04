# ERes2Net

## Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 512, 8 gpus(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF(p-target=0.01)

# Largin-margin-finetune
- Feature: 80-dim fbank, mean normalization
- Training: lr [2e-5, 1e-4], batch_size 128, 4 gpus(Tesla V100), additive angular margin=0.5

## Voxceleb Results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| ERes2Net-base | 4.6M | 0.97  |  0.090 |
| ERes2Net-base-lm | 4.6M | 0.84  |  0.086 |

## pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- Voxceleb: [speech_eres2net_sv_en_voxceleb_16k](https://modelscope.cn/models/damo/speech_eres2net_sv_en_voxceleb_16k/summary)
- 200k labeled speakers: [speech_eres2net_sv_zh-cn_16k-common](https://modelscope.cn/models/damo/speech_eres2net_sv_zh-cn_16k-common/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# ERes2Net trained on VoxCeleb
model_id=damo/speech_eres2net_sv_en_voxceleb_16k
# ERes2Net trained on 200k labeled speakers
model_id=damo/speech_eres2net_sv_zh-cn_16k-common
# Run inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

## Citations
If you are using ERes2Net model in your research, please cite: 
```BibTeX
@article{eres2net,
  title={An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification},
  author={Yafeng Chen, Siqi Zheng, Hui Wang, Luyao Cheng, Qian Chen, Jiajun Qi},
  booktitle={Interspeech 2023},
  year={2023},
  organization={IEEE}
}
```
