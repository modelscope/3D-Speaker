
<p align="center">
    <br>
    <img src="docs/images/3D-Speaker.png" width="400"/>
    <br>
<p>
    
<div align="center">

<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->
![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-Linux-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.8-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=1.10-blue"></a>
    
</div>
    
<strong>3D-Speaker</strong> is an open-source toolkit for single- and multi-modal speaker verification, speaker recognition, and speaker diarization. All pre-trained models are accessible on [ModelScope](https://www.modelscope.cn/models).

## News
- [2023.4] [RDINO](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/sv-rdino/voxceleb) training recipes on [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) released. RDINO is a self-supervised learning framework in speaker verification aiming to alleviate model collapse in non-contrastive methods. It contains teacher and student network with an identical architecture but different parameters. Two regularization terms are proposed in RDINO, namely diversity regularization and redundancy elimination regularization. RDINO achieve 3.05% EER and 0.220 MinDCF in VoxCeleb using single-stage self-supervised training.
- [2023.4] [CAM++](https://www.modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary) pretrained model released, trained on a Mandarin dataset of 200k labeled speakers. 
- [2023.4] [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/sv-cam++/voxceleb) training recipe on [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) released. CAM++ is a fast and efficient speaker embedding extractor based on a densely connected time-delay neural network (D-TDNN). It adopts a novel multi-granularity pooling method to conduct context-aware masking. CAM++ achieves an EER of 0.73% in Voxceleb and 6.78% in CN-Celeb, outperforming other mainstream speaker embedding models such as ECAPA-TDNN and ResNet34, while having lower computational cost and faster inference speed.

## To be expected
- [2023.5] Releasing ERes2Net (Enhanced Res2Net) training framework.
- [2023.5] Releasing ERes2Net model trained on over 100k labeled speakers.

## Installation
``` sh
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git && cd 3D-Speaker
conda create -n 3D-Speaker python=3.8
conda activate 3D-Speaker
pip install -r requirements.txt
```

## Pretrained model
3D-Speaker shares pretrained models on [ModelScope](https://www.modelscope.cn/models).
| Task | Dataset | Model | Performance |
|:-----:|:------:|:------:|:------:|
| speaker verification | VoxCeleb | [CAM++](https://modelscope.cn/models/damo/speech_campplus_sv_en_voxceleb_16k/summary) | EER = 0.73% |
| speaker verification | VoxCeleb | [RDINO](https://modelscope.cn/models/damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k/summary) | EER = 3.24% |

Here is another simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and generates embeddings.
``` sh
# install modelscope
pip install modelscope
# extract embeddings based on pretrained models
# CAM++ trained on VoxCeleb
model_id=damo/speech_campplus_sv_en_voxceleb_16k
model_revision=v1.0.2
python speakerlab/bin/infer_sv.py --model_id $model_id --model_revision $model_revision --wav_path $wav_path
# CAM++ trained on 200k labeled speakers
model_id=damo/speech_campplus_sv_zh-cn_16k-common
model_revision=v1.0.0
python speakerlab/bin/infer_sv.py --model_id $model_id --model_revision $model_revision --wav_path $wav_path
# RDINO on VoxCeleb
model_id=damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
model_revision=v1.0.1
python speakerlab/bin/infer_sv.py --model_id $model_id --model_revision $model_revision --wav_path $wav_path
```

## License
3D-Speaker is released under the [Apache License 2.0](LICENSE).

## Acknowledge
3D-Speaker contains third-party components and code modified from some open-source repos, including:

- [speechbrain](https://github.com/speechbrain/speechbrain)
- [wespeaker](https://github.com/wenet-e2e/wespeaker)
- [D-TDNN](https://github.com/yuyq96/D-TDNN)
- [dino](https://github.com/facebookresearch/dino)
- [vicreg](https://github.com/facebookresearch/vicreg)

## Contact
If you have any comment or question about 3D-Speaker, please contact us by
- email: chenyafeng.cyf@alibaba-inc.com, tongmu.wh@alibaba-inc.com

## Citations
```BibTeX
@inproceedings{rdino,
  title={Pushing the limits of self-supervised speaker verification using regularized distillation framework},
  author={Yafeng Chen and Siqi Zheng and Hui Wang and Luyao Cheng and Qian Chen},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
@article{cam++,
  title={CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking},
  author={Hui Wang and Siqi Zheng and Yafeng Chen and Luyao Cheng and Qian Chen},
  journal={arXiv preprint arXiv:2303.00332},
  year={2023}
}
```
