## ERes2Net

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 256, 4 gpu(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF

### CNCeleb results
- Train set: CNCeleb-dev + CNCeleb2, 2973 speakers
- Test set: CNCeleb-eval

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| CAM++ | 7.18M  | 6.78 | 0.393 |
| ERes2net-base | 6.61M  | 6.69 | 0.388 |
| ERes2net-large | 22.46M  | 6.17 | 0.372 |
| ERes2netV2-lm | 17.8M  | 6.14 | 0.370 |

### Pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- ERes2net-base: [speech_eres2net_base_sv_zh-cn_cnceleb_16k](https://modelscope.cn/models/damo/speech_eres2net_base_sv_zh-cn_cnceleb_16k/summary)
- ERes2Net-large: [speech_eres2net_large_sv_zh-cn_cnceleb_16k](https://modelscope.cn/models/damo/speech_eres2net_large_sv_zh-cn_cnceleb_16k/summary)
- 200k labeled speakers: [speech_eres2net_sv_zh-cn_16k-common](https://modelscope.cn/models/damo/speech_eres2net_sv_zh-cn_16k-common/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# ERes2Net trained on CNCeleb
model_id=damo/speech_eres2net_base_sv_zh-cn_cnceleb_16k
# ERes2Net trained on 200k labeled speakers
model_id=damo/speech_eres2net_sv_zh-cn_16k-common
# Run inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

### Citations
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
