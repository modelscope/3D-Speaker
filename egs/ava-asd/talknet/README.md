# Active Speaker Detection

## Introduction
Active speaker detection is designed to detect who is speaking in a visual scene with potentially multiple speakers. This recipe provides training methods using the TalkNet model, adapted from the original repository https://github.com/TaoRuijie/TalkNet-ASD. Compared to the original repository, this adapted version features multi-process data processing and multi-GPU training and evaluation, resulting in a more streamlined pipeline and accelerated performance.

## Usage
``` sh
pip install -r requirements.txt
bash run.sh
# Make sure ffmpeg is available in your environment.
# It can be installed using:
sudo apt-get update
sudo apt-get install ffmpeg
# or using
conda install ffmpeg
```

## Evaluation
The evaluation results on the AVA-ActiveSpeaker val dataset using the mAP metric.
| Test | mAp |
|:-----:|:------:|
|AVA-ActiveSpeaker|92.0%|


## Citation
```BibTeX
@inproceedings{tao2021someone,
  title={Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection},
  author={Tao, Ruijie and Pan, Zexu and Das, Rohan Kumar and Qian, Xinyuan and Shou, Mike Zheng and Li, Haizhou},
  booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
  pages = {3927–3935},
  year={2021}
}
```
