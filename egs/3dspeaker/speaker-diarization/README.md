# Speaker diarization

## Introduction
This recipe offers speaker diarization methods that address the problem of "who spoke when". It provides two pipelines: audio-only and multimodal diarization. The audio-only diarization comprises multiple modules, including voice activity detection, speech segmentation, speaker embedding extraction, and speaker clustering. The multimodal approach fuses audio and video image input to produce more accurate results.

## Usage
``` sh
pip install -r requirements.txt
# audio-only diarization
bash run_audio.sh
# multimodal diarization
bash run_video.sh
```

## Evaluation
The results of audio-only diarization pipeline on two-speaker and multi-speaker datasets using the DER metric.
| Test | DER |
|:-----:|:------:|
|Two-speaker|4.7%|
|Muti-speaker(2-10)|8.0%|

The comparison results of two diarization pipelines on a multi-person conversation video dataset using the DER metric.
| Pipeline | DER |
|:-----:|:------:|
|Audio-only diarization|5.0%|
|Multimodal diarization|3.7%|


## Limitations
- It cannot address the issue of overlapped speech, where multiple speakers speak at the same time. 
- It may not perform well when the audio duration is too short (less than 30 seconds) and when the number of speakers is too large
- The final accuracy is highly dependent on the performance of each modules. Therefore, using pretrained models that are more aligned with the test scenario may result in higher accuracy.
