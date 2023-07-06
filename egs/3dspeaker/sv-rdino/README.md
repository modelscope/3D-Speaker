# RDINO

## Training config
- Feature: 80-dim fbank
- Training: batch_size 52 * 4, 4 gpu(Tesla V100)
- Metrics: EER(%), MinDCF(p-target=0.05)

## 3D-Speaker Results
- Train set: 3D-Speaker-train
- Test set: Cross-Device, Cross-Distance, Cross-Dialect

 Performance of systems (EER) on different trials.
| Model | Params | Cross-Device | Cross-Distance | Cross-Dialect |
|:-----:|:------:| :------:|:------:|:------:|
| RDINO | 45.44M | 20.41% | 21.92% | 25.53% |


## pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).<br>
Please look forward it.

## Citations
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
