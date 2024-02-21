## ECAPA-TDNN

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

### Pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- ECAPA-TDNN: [speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k](https://modelscope.cn/models/damo/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# ECAPA-TDNN trained on 3D-Speaker
model_id=damo/speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k
# Run inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

### Citations
If you are using ECAPA-TDNN model in your research, please cite: 
```BibTeX
@article{desplanques2020ecapa,
  title={Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  journal={arXiv preprint arXiv:2005.07143},
  year={2020}
}
```
