# CAM++

## Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.0001, 0.1], batch_size 256, 4 gpu(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF

## 3D-Speaker Results
- Train set: 3D-Speaker-train
- Test set: 3D-Speaker-test

| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| ECAPA-TDNN | 20.8M | 8.87% | 12.26% | 14.53% |
| CAM++ Base | 7.2M | 7.75% | 11.29% | 13.44% |

## pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).<br>
Please look forward it.

## Citations
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
