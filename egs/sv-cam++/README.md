# CAM++

## Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.0001, 0.1], batch_size 256, 4 gpu(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF

## Voxceleb Results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| CAM++ | 7.18M  | 0.73 | 0.0911 |

## pretrained model
Voxceleb: [ModelScope](https://www.modelscope.cn/models)：[speech_campplus_sv_en_voxceleb_16k](https://modelscope.cn/models/damo/speech_campplus_sv_en_voxceleb_16k/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and generates embeddings.
``` sh
# install modelscope
pip install modelscope
# extract embeddings from the pretrained models
# CAM++ on VoxCeleb
model_id=damo/speech_campplus_sv_en_voxceleb_16k
model_revision=v1.0.2
python speakerlab/bin/infer_sv.py --model_id $model_id --model_revision $model_revision --wav_path $wav_path
```
