## VoxCeleb

### Running experiments
``` sh
# Speaker verification: ERes2Net on VoxCeleb
cd egs/voxceleb/sv-eres2net/
bash run.sh
# Speaker verification: CAM++ on VoxCeleb
cd egs/voxceleb/sv-cam++/
bash run.sh
# Speaker verification: ECAPA-TDNN on VoxCeleb
cd egs/voxceleb/sv-ecapa/
bash run.sh
# Speaker verification: ResNet on VoxCeleb
cd egs/voxceleb/sv-resnet/
bash run.sh
# Speaker verification: Res2Net on VoxCeleb
cd egs/voxceleb/sv-res2net/
bash run.sh
# Self-supervised speaker verification: RDINO on VoxCeleb
cd egs/voxceleb/sv-rdino/
bash run.sh
```

 EER(%) of systems on VoxCeleb-O, VoxCeleb-E and VoxCeleb-H.
| Model | Params | VoxCeleb-O | VoxCeleb-E | VoxCeleb-H |
|:-----:|:------:| :------:|:------:|:------:|
| TDNN | 4.34 M | 2.22 | 2.17 | 3.63 |
| Res2Net | 4.03 M | 1.56 | 1.41 | 2.48 |
| ResNet34 | 6.34 M | 1.05 | 1.11 | 1.99 |
| ERes2Net-base | 6.61 M | 0.96 | 1.06 | 1.96 |
| ECAPA-TDNN | 20.8 M | 0.86 | 0.97 | 1.90 |
| CAM++ | 7.2 M | 0.73 | 0.89 | 1.76 |
| ERes2Net-large-lm | 22.46 M | 0.52 | 0.75 | 1.44 |
| ERes2NetV2-lm | 17.8M | 0.61  |  0.76 | 1.45 |

### Inference using pretrained models from Modelscope
All pretrained models are released on [Modelscope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

``` sh
# Install modelscope
pip install modelscope
# CAM++ trained on VoxCeleb
model_id=iic/speech_campplus_sv_en_voxceleb_16k
# ERes2Net-base trained on VoxCeleb
model_id=iic/speech_eres2net_sv_en_voxceleb_16k
# ERes2Net-large trained on VoxCeleb
model_id=iic/speech_eres2net_large_sv_en_voxceleb_16k
# resnet trained on VoxCeleb
model_id=iic/speech_resnet_sv_en_voxceleb_16k
# res2net trained on VoxCeleb
model_id=iic/speech_res2net_sv_en_voxceleb_16k
# Run CAM++ or ERes2Net inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path

# RDINO trained on VoxCeleb
model_id=iic/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
# Run rdino inference
python speakerlab/bin/infer_sv_rdino.py --model_id $model_id --wavs $wav_path
