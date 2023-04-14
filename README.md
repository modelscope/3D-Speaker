# 3D-Speaker

<strong>3D-Speaker</strong> is an open-source toolkit for single- and multi-modal speaker verification, speaker recognition, and speaker diarization. All pretrained models can be accessed in [ModelScope](https://www.modelscope.cn/models).


## News
- [2023.4] [CAM++](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/sv-cam++/voxceleb) training recipes on [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) released. CAM++ is a fast and efficient speaker embedding extractor based on a densely connected time-delay neural network (D-TDNN), adopting a novel multi-granularity pooling to conduct context-aware masking. CAM++ achieve 0.73% EER in Voxceleb and 6.78% EER in CN-Celeb, outperforming other main-stream speaker embedding models such as ECAPA-TDNN and ResNet32 with lower computational cost and faster inference speed.


## To be expected
- [2023.4] Releasing RDINO model.
- [2023.5] Releasing CAM++ model trained on over 100k labeled speakers. 

## Installation
``` sh
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git && cd 3D-Speaker
conda create -n 3D-Speaker python=3.8
conda activate 3D-Speaker
pip install -r requirements.txt
```

## Pretrained model
3D-Speaker sharing the pretrained models on [ModelScope](https://www.modelscope.cn/models)
| Task | Dataset | Model | Performance |
|:-----:|:------:|:------:|:------:|
| speaker verification | VoxCeleb | [CAM++](https://modelscope.cn/models/damo/speech_campplus_sv_en_voxceleb_16k/summary) | EER=0.73% |

Here is another simple example to directly extract embeddings. It will download the pretrained model from [ModelScope](https://www.modelscope.cn/models) and generate embeddings.
``` sh
# install modelscope
pip install modelscope
# extract embeddings from the pretrained models
# CAM++ on VoxCeleb
model_id=damo/speech_campplus_sv_en_voxceleb_16k
model_revision=v1.0.2
python speakerlab/bin/infer_sv.py --model_id $model_id --model_revision $model_revision --wav_path $wav_path
```

## License
3D-Speaker is released under the [Apache License 2.0](LICENSE).

## Acknowledge
3D-Speaker contains third-party components and code modified from some open source repos, including:

- [speechbrain](https://github.com/speechbrain/speechbrain)
- [wespeaker](https://github.com/wenet-e2e/wespeaker)
- [D-TDNN](https://github.com/yuyq96/D-TDNN)

## Contact
If you have any comment or question about 3D-Speaker, please contact us by
- email: chenyafeng.cyf@alibaba-inc.com, tongmu.wh@alibaba-inc.com

## Citations
```BibTeX
@article{cam++,
  title={CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking},
  author={Hui Wang and Siqi Zheng and Yafeng Chen and Luyao Cheng and Qian Chen},
  journal={arXiv preprint arXiv:2303.00332},
  year={2023}
}
```
