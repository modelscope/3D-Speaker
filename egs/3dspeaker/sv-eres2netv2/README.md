## ERes2Net

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 512, 8 gpu(Tesla V100), additive angular margin, speaker embeddding=192
- Metrics: EER(%), MinDCF(p-target=0.01)

### 3D-Speaker results
- Train set: 3D-Speaker-train
- Test set: 3D-Speaker-test

| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| ECAPA-TDNN | 20.8M | 8.87% | 12.26% | 14.53% |
| ERes2Net-base | 6.61M | 7.06% | 9.95% | 12.76% |
| ERes2Net-large | 22.46M| 6.55% | 9.45% | 11.01% |
| ERes2NetV2-lm | 17.8M | 6.52% | 8.88% | 11.34% |


### Pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- ERes2net-base: [speech_eres2net_base_sv_zh-cn_3dspeaker_16k](https://modelscope.cn/models/damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k/summary)
- ERes2Net-large: [damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k](https://modelscope.cn/models/damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/summary)
- 200k labeled speakers: [speech_campplus_sv_zh-cn_16k-common](https://www.modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# ERes2Net trained on 3D-Speaker
model_id=damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k
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
