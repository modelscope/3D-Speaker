## RDINO

### Training config
- Feature: 80-dim fbank
- Training: batch_size 52 * 4, 4 gpu(Tesla V100)
- Metrics: EER(%), MinDCF(p-target=0.05)

### Voxceleb results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb1-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| RDINO perforance | 45.4M | 3.16  |  0.223 |

Note: The original checkpoint is uploaded to ModelScope. The batchsize would affect the learning rate and the number of iterations. It could get the same or similar results if the parameters are unchanged.

### Pretrained model in Voxceleb
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- Voxceleb: [damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k](https://www.modelscope.cn/models/damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and generates embeddings.
``` sh
# Install modelscope
pip install modelscope
# RDINO trained on VoxCeleb
model_id=damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
# Run inference
python speakerlab/bin/infer_sv_rdino.py --model_id $model_id --wavs $wav_path
```

### Citations
If you are using RDINO model in your research, please cite: 
```BibTeX
@inproceedings{chen2023pushing,
  title={Pushing the limits of self-supervised speaker verification using regularized distillation framework},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and Chen, Qian},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
