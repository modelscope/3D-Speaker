## CN-Celeb

Dataset introduction and download address: [CN-Celeb](http://cnceleb.org/) <br>
Related paper address: [CN-Celeb](https://arxiv.org/pdf/2012.12468.pdf)

### Running experiments
``` sh
# Speaker verification: ERes2Net on CN-Celeb
cd egs/cnceleb/sv-eres2net/
bash run.sh
# Speaker verification: CAM++ on CN-Celeb
cd egs/cnceleb/sv-cam++/
bash run.sh
# Speaker verification: ECAPA-TDNN on CN-Celeb
cd egs/cnceleb/sv-ecapa/
bash run.sh
# Speaker verification: ResNet on CN-Celeb
cd egs/cnceleb/sv-resnet/
bash run.sh
# Speaker verification: Res2Net on CN-Celeb
cd egs/cnceleb/sv-res2net/
bash run.sh
# Self-supervised speaker verification: RDINO on CN-Celeb
cd egs/cnceleb/sv-rdino/
bash run.sh
```
Performance of systems on CN-Celeb-test.
| Model | Params | EER(%) | MinDCF |
|:-----:|:------:| :------:| :------:|
| ECAPA-TDNN | 20.8 M | 8.01 | 0.445 |
| Res2Net | 4.03 M | 7.96 | 0.452 |
| ResNet34 | 6.34 M | 6.92 | 0.421 |
| CAM++ | 7.18 M | 6.78 | 0.393 |
| ERes2Net-base | 6.61 M | 6.69 | 0.388 |
| ERes2Net-large | 22.46 M | 6.17 | 0.372 |
| ERes2NetV2 | 17.85 M | 6.14 | 0.370 |
| RDINO | 45.44 M | 17.07 | 0.602 |

### Inference using pretrained models from Modelscope
All pretrained models will be released on [Modelscope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio). <br>

``` sh
# Install modelscope
pip install modelscope
# CAM++ trained on CN-Celeb
model_id=iic/speech_campplus_sv_cn_cnceleb_16k
# CAM++ trained on 200k labeled speakers
model_id=iic/speech_campplus_sv_zh-cn_16k-common
# ERes2Net-base trained on CN-Celeb
model_id=iic/speech_eres2net_base_sv_zh-cn_cnceleb_16k
# ERes2Net-large trained on CN-Celeb
model_id=iic/speech_eres2net_large_sv_zh-cn_cnceleb_16k
# ERes2Net trained on 200k labeled speakers
mode_id=iic/speech_eres2net_sv_zh-cn_16k-common
# Run CAM++ or ERes2Net inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path

# RDINO trained on CN-Celeb
model_id=iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k
# Run RDINO inference
python speakerlab/bin/infer_sv_rdino.py --model_id $model_id --wavs $wav_path
```


