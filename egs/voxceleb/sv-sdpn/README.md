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
| SDPN perforance | 57.24M | 1.80  |  0.139 |

### Pretrained model in Voxceleb
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- Voxceleb: [iic/speech_sdpn_ecapa_tdnn_sv_en_voxceleb_16k](https://modelscope.cn/models/iic/speech_sdpn_ecapa_tdnn_sv_en_voxceleb_16k)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and generates embeddings.
``` sh
# Install modelscope
pip install modelscope
# SDPN trained on VoxCeleb
model_id=iic/speech_sdpn_ecapa_tdnn_sv_en_voxceleb_16k
# Run inference
python speakerlab/bin/infer_sv_ssl.py --model_id $model_id
```

### Citations
If you are using SDPN model in your research, please cite: 
```BibTeX
@article{chen2024sdpn,
  title={Self-Distillation Prototypes Network: Learning Robust Speaker Representations without Supervision},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and others},
  booktitle={ICASSP},
  year={2025}
}
