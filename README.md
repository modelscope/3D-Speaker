
<p align="center">
    <br>
    <img src="docs/images/3D-Speaker-logo.png" width="400"/>
    <br>
<p>
    
<div align="center">
    
<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->
![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-Linux-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.8-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=1.10-blue"></a>
    
</div>
    
<strong>3D-Speaker</strong> is an open-source toolkit for single- and multi-modal speaker verification, speaker recognition, and speaker diarization. All pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio). Furthermore, we present a large-scale speech corpus also called [3D-Speaker](https://3dspeaker.github.io/) to facilitate the research of speech representation disentanglement.

## Quickstart
### Install 3D-Speaker
``` sh
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git && cd 3D-Speaker
conda create -n 3D-Speaker python=3.8
conda activate 3D-Speaker
pip install -r requirements.txt
```
### Running experiments
``` sh
# Speaker verification: ERes2Net on 3D-Speaker dataset
cd egs/3dspeaker/sv-eres2net/
bash run.sh
# Speaker verification: ERes2NetV2 on 3D-Speaker dataset
cd egs/3dspeaker/sv-eres2netv2/
bash run.sh
# Speaker verification: CAM++ on 3D-Speaker dataset
cd egs/3dspeaker/sv-cam++/
bash run.sh
# Speaker verification: ECAPA-TDNN on 3D-Speaker dataset
cd egs/3dspeaker/sv-ecapa/
bash run.sh
# Self-supervised speaker verification: RDINO on 3D-Speaker dataset
cd egs/3dspeaker/sv-rdino/
bash run.sh
# Self-supervised speaker verification: SDPN on VoxCeleb dataset
cd egs/voxceleb/sv-sdpn/
bash run.sh
# Audio and multimodal Speaker diarization:
cd egs/3dspeaker/speaker-diarization/
bash run_audio.sh
bash run_video.sh
# Language identification
cd egs/3dspeaker/language-idenitfication
bash run.sh
```
### Inference using pretrained models from Modelscope
All pretrained models are released on [Modelscope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

``` sh
# Install modelscope
pip install modelscope
# ERes2Net trained on 200k labeled speakers
model_id=iic/speech_eres2net_sv_zh-cn_16k-common
# ERes2NetV2 trained on 200k labeled speakers
model_id=iic/speech_eres2netv2_sv_zh-cn_16k-common
# CAM++ trained on 200k labeled speakers
model_id=iic/speech_campplus_sv_zh-cn_16k-common
# Run CAM++ or ERes2Net inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path

# RDINO trained on VoxCeleb
model_id=iic/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
# Run rdino inference
python speakerlab/bin/infer_sv_rdino.py --model_id $model_id --wavs $wav_path
```

## Overview of Content

- **Supervised Speaker Verification**
  - [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-cam%2B%2B), [ERes2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-eres2net), [ERes2NetV2](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-eres2netv2), [ECAPA-TDNN](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-ecapa), [ResNet](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-resnet) and [Res2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-res2net) training recipes on [3D-Speaker](https://3dspeaker.github.io/).

  - [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/voxceleb/sv-cam%2B%2B), [ERes2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/voxceleb/sv-eres2net), [ERes2NetV2](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-eres2netv2), [ECAPA-TDNN](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-ecapa), [ResNet](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-resnet) and [Res2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-res2net) training recipes on [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/). 

  - [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-cam%2B%2B), [ERes2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-eres2net), [ERes2NetV2](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-eres2netv2), [ECAPA-TDNN](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-ecapa), [ResNet](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-resnet) and [Res2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-res2net) training recipes on [CN-Celeb](http://cnceleb.org/).

- **Self-supervised Speaker Verification**
  - [RDINO](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/voxceleb/sv-rdino) and [SDPN](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-sdpn) training recipes on [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
    
  - [RDINO](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/sv-rdino) training recipes on [3D-Speaker](https://3dspeaker.github.io/).

  - [RDINO](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-rdino) training recipes on [CN-Celeb](http://cnceleb.org/).

- **Speaker Diarization**
  - [Speaker diarization](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/3dspeaker/speaker-diarization) inference recipes which comprise multiple modules, including voice activity detection, speech segmentation, speaker embedding extraction, and speaker clustering. 

- **Language Identification**
  - [Language identification](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/language-identification) training recipes on [3D-Speaker](https://3dspeaker.github.io/).

- **3D-Speaker Dataset**
  - Dataset introduction and download address: [3D-Speaker](https://3dspeaker.github.io/) <br>
  - Related paper address: [3D-Speaker](https://arxiv.org/pdf/2306.15354.pdf)


## What‘s new :fire:
- [2024.5] Releasing [X-vector](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-xvector) model on VoxCeleb datasets.
- [2024.5] Releasing [SDPN](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-sdpn) model training and inference recipes for VoxCeleb.
- [2024.5] Releasing [visual module](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/ava-asd/talknet) and [semantic module](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/semantic_speaker/bert) training recipes.
- [2024.4] Releasing [ONNX Runtime](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/runtime/onnxruntime) and the relevant scripts for inference.
- [2024.4] Releasing [ERes2NetV2](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-eres2netv2) model with lower parameters and faster inference speed on VoxCeleb datasets.
- [2024.2] Releasing [language identification](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/language-identification) integrating phonetic information recipes for more higher recognition accuracy.
- [2024.2] Releasing [multimodal diarization](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/3dspeaker/speaker-diarization) recipes which fuses audio and video image input to produce more accurate results.
- [2024.1] Releasing [ResNet34](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-resnet) and [Res2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-res2net) model training and inference recipes for 3D-Speaker, VoxCeleb and CN-Celeb datasets.
- [2024.1] Releasing [large-margin finetune recipes](https://github.com/alibaba-damo-academy/3D-Speaker/blob/main/egs/voxceleb/sv-eres2net/run.sh) in speaker verification and adding [diarization recipes](https://github.com/alibaba-damo-academy/3D-Speaker/blob/main/egs/3dspeaker/speaker-diarization/run.sh). 
- [2023.11] [ERes2Net-base](https://modelscope.cn/models/damo/speech_eres2net_base_200k_sv_zh-cn_16k-common/summary) pretrained model released, trained on a Mandarin dataset of 200k labeled speakers.
- [2023.10] Releasing [ECAPA model](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-ecapa) training and inference recipes for three datasets.
- [2023.9] Releasing [RDINO](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/cnceleb/sv-rdino) model training and inference recipes for [CN-Celeb](http://cnceleb.org/).
- [2023.8] Releasing [CAM++](https://modelscope.cn/models/damo/speech_campplus_sv_cn_cnceleb_16k/summary), [ERes2Net-Base](https://modelscope.cn/models/damo/speech_eres2net_base_sv_zh-cn_cnceleb_16k/summary) and [ERes2Net-Large](https://modelscope.cn/models/damo/speech_eres2net_large_sv_zh-cn_cnceleb_16k/summary) benchmarks in [CN-Celeb](http://cnceleb.org/).
- [2023.8] Releasing [ERes2Net](https://modelscope.cn/models/damo/speech_eres2net_base_lre_en-cn_16k/summary) annd [CAM++](https://modelscope.cn/models/damo/speech_campplus_lre_en-cn_16k/summary) in language identification for Mandarin and English. 
- [2023.7] Releasing [CAM++](https://modelscope.cn/models/damo/speech_campplus_sv_zh-cn_3dspeaker_16k/summary), [ERes2Net-Base](https://modelscope.cn/models/damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k/summary), [ERes2Net-Large](https://modelscope.cn/models/damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/summary) pretrained models trained on [3D-Speaker](https://3dspeaker.github.io/).
- [2023.7] Releasing [Dialogue Detection](https://modelscope.cn/models/damo/speech_bert_dialogue-detetction_speaker-diarization_chinese/summary) and [Semantic Speaker Change Detection](https://modelscope.cn/models/damo/speech_bert_semantic-spk-turn-detection-punc_speaker-diarization_chinese/summary) in speaker diarization.
- [2023.7] Releasing [CAM++](https://modelscope.cn/models/damo/speech_campplus_lre_en-cn_16k/summary) in language identification for Mandarin and English.
- [2023.6] Releasing [3D-Speaker](https://3dspeaker.github.io/) dataset and its corresponding benchmarks including [ERes2Net](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/3dspeaker/sv-eres2net), [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/3dspeaker/sv-cam%2B%2B) and [RDINO](https://github.com/alibaba-damo-academy/3D-Speaker/tree/3dspeaker/egs/3dspeaker/sv-rdino).
- [2023.5] [ERes2Net](https://modelscope.cn/models/damo/speech_eres2net_sv_zh-cn_16k-common/summary) pretrained model released, trained on a Mandarin dataset of 200k labeled speakers.
- [2023.4] [CAM++](https://www.modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary) pretrained model released, trained on a Mandarin dataset of 200k labeled speakers.

## To be expected :fire:
- [2024.5] Releasing asnormalization in speaker verification.
- [2024.5] Supporting more effective models.

## Contact
If you have any comment or question about 3D-Speaker, please contact us by
- email: {chenyafeng.cyf, zsq174630, tongmu.wh, shuli.cly}@alibaba-inc.com

## License
3D-Speaker is released under the [Apache License 2.0](LICENSE).

## Acknowledge
3D-Speaker contains third-party components and code modified from some open-source repos, including: <br>
[Speechbrain](https://github.com/speechbrain/speechbrain), [Wespeaker](https://github.com/wenet-e2e/wespeaker), [D-TDNN](https://github.com/yuyq96/D-TDNN), [DINO](https://github.com/facebookresearch/dino), [Vicreg](https://github.com/facebookresearch/vicreg), [TalkNet-ASD
](https://github.com/TaoRuijie/TalkNet-ASD), [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)


## Citations
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```BibTeX
@article{chen2024eres2netv2,
  title={ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational Efficiency},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and Chen, Qian and Zhang, Shiliang and Li, Junjie},
  journal={arXiv preprint arXiv:2406.02167},
  year={2024}
}
@article{chen20243d,
  title={3D-Speaker-Toolkit: An Open Source Toolkit for Multi-modal Speaker Verification and Diarization},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and others},
  journal={arXiv preprint arXiv:2403.19971},
  year={2024}
}
@inproceedings{zheng20233d,
  title={3D-Speaker: A Large-Scale Multi-Device, Multi-Distance, and Multi-Dialect Corpus for Speech Representation Disentanglement},
  author={Siqi Zheng, Luyao Cheng, Yafeng Chen, Hui Wang and Qian Chen},
  url={https://arxiv.org/pdf/2306.15354.pdf},
  year={2023}
}
@inproceedings{wang2023cam++,
  title={CAM++: A Fast and Efficient Network For Speaker Verification Using Context-Aware Masking},
  author={Wang, Hui and Zheng, Siqi and Chen, Yafeng and Cheng, Luyao and Chen, Qian},
  year={2023},
  booktitle={INTERSPEECH}
}
@inproceedings{chen2023enhanced,
  title={An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and Chen, Qian and Qi, Jiajun},
  year={2023},
  booktitle={INTERSPEECH}
}
@inproceedings{chen2023pushing,
  title={Pushing the limits of self-supervised speaker verification using regularized distillation framework},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and Chen, Qian},
  booktitle={ICASSP 2023},
  year={2023}
}
```
