# RDINO

## Training config
- Feature: 80-dim fbank
- Training: batch_size 52 * 4, 4 gpu(Tesla V100)
- Metrics: EER(%), MinDCF(p-target=0.05)

## Voxceleb Results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb1-O

| Model | EER(%) | MinDCF |
|:-----:|:------:|:------:|
| RDINO performance in the paper |  3.24  |  0.252 |
| RDINO perforance reproduced with this code |  3.05  |  0.220 |

Note: The original checkpoint is uploaded to ModelScope. The batchsize would affect the learning rate and the number of iterations. It could get the same or similar results if the parameters are unchanged.

## Pretrained model in Voxceleb
[ModelScope](https://www.modelscope.cn/home)：[damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k](https://www.modelscope.cn/models/damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and generates embeddings.
``` sh
# Install modelscope
pip install modelscope
# RDINO trained on VoxCeleb
model_id=damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
# Run inference
python speakerlab/bin/infer_sv_rdino.py --model_id $model_id --wavs $wav_path
```
