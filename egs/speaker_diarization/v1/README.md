# Speaker diarization

## Introduction
This recipe offers a speaker diarization pipeline that addresses the problem of "who spoke when". It comprises multiple modules, including voice activity detection, speech segmentation, speaker embedding extraction, and speaker clustering.

## Modules
- Voice activity detection model: [speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
- Speaker embedding model: [speech_campplus_sv_zh-cn_16k-common](https://www.modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary)
- Speaker clustering. 
  - Spectral Clustering: Suitable for medium-length audio (<30min) with relatively few speakers (<6).
  - UMAP-HDBSCAN: Suitable for long-length audio (>30min) with a relatively large number of speakers (>5).

## Usage
First prepare a example wav list:
``` sh
mkdir examples
wget "https://modelscope.cn/api/v1/models/damo/speech_campplus_speaker-diarization_common/repo?Revision=master&FilePath=examples/2speakers_example.wav" -O examples/2speakers_example.wav
find examples -name "*.wav" > examples/wav.list
```
Then run:
``` sh
pip install -r requirements.txt
bash run.sh
```

## Evaluation
The results are evaluated on two-speaker and multi-speaker Mandarin datasets using the DER metric.
| Test | DER |
|:-----:|:------:|
|Two-speaker|4.7%|
|Muti-speaker(2-10)|8.0%|


## Limitations
- It cannot address the issue of overlapped speech, where multiple speakers speak at the same time. 
- It may not perform well when the audio duration is too short (less than 30 seconds) and when the number of speakers is too large (more than 10). 
- The final accuracy is highly dependent on the performance of each modules. Therefore, using pretrained models that are more aligned with the test scenario may result in higher accuracy.
