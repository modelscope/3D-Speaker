# VoxCeleb

### Running experiments
``` sh
# Speaker verification: ERes2Net on VoxCeleb
cd egs/voxceleb/sv-eres2net/
bash run.sh
# Speaker verification: CAM++ on VoxCeleb
cd egs/voxceleb/sv-cam++/
bash run.sh
# Self-supervised speaker verification: RDINO on VoxCeleb
cd egs/voxceleb/sv-rdino/
bash run.sh
```
### Inference using pretrained models from Modelscope
All pretrained models are released on [Modelscope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

``` sh
# Install modelscope
pip install modelscope
# CAM++ trained on VoxCeleb
model_id=damo/speech_campplus_sv_en_voxceleb_16k
# CAM++ trained on 200k labeled speakers
model_id=damo/speech_campplus_sv_zh-cn_16k-common
# ERes2Net trained on VoxCeleb
model_id=damo/speech_eres2net_sv_en_voxceleb_16k
# ERes2Net trained on 200k labeled speakers
model_id=damo/speech_eres2net_sv_zh-cn_16k-common
# Run CAM++ or ERes2Net inference
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs $wav_path

# RDINO trained on VoxCeleb
model_id=damo/speech_rdino_ecapa_tdnn_sv_en_voxceleb_16k
# Run rdino inference
python speakerlab/bin/infer_sv_rdino.py --model_id $model_id --wavs $wav_path