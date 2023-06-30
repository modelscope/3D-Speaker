# ERes2Net

## Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 512, 8 gpu(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF(p-target=0.01)

## 3D-Speaker Results
- Train set: 3D-Speaker-train
- Test set: 3D-Speaker-test

| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| ECAPA-TDNN | 20.8M | 8.87% | 12.26% | 14.53% |
| ERes2Net Base | 4.6M | 7.21% | 10.18% | 12.52% |
| ERes2Net Large | 18.3M | 6.89% | 10.36% | 11.97% |

## pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).<br>
Please look forward it.

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
