## CAM++

### Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.0001, 0.1], batch_size 256, 4 gpu(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF

### 3D-Speaker results
- Train set: 3D-Speaker-train
- Test set: 3D-Speaker-test

| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| ECAPA-TDNN | 20.8M | 8.87% | 12.26% | 14.53% |
| CAM++ | 7.18M | 7.75% | 11.29% | 13.44% |

### Pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- CAM++: [speech_campplus_sv_zh-cn_3dspeaker_16k](https://modelscope.cn/models/damo/speech_campplus_sv_zh-cn_3dspeaker_16k/summary)

- 200k labeled speakers: [speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# CAM++ trained on 3D-Speaker
model_id=damo/speech_campplus_sv_zh-cn_3dspeaker_16k
# CAM++ trained on 200k labeled speakers
model_id=damo/speech_campplus_sv_zh-cn_16k-common
# Run inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

### Citations
If you are using CAM++ model in your research, please cite: 
```BibTeX
@article{cam++,
  title={CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking},
  author={Hui Wang and Siqi Zheng and Yafeng Chen and Luyao Cheng and Qian Chen},
  booktitle={Interspeech 2023},
  year={2023},
  organization={IEEE}
}
```
