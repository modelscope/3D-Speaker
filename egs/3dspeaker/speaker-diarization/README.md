# Speaker diarization

## Introduction
This recipe offers speaker diarization methods that address the problem of "who spoke when". It provides two pipelines: audio-only and multimodal diarization. The audio-only diarization comprises multiple modules, including overlap detection[optional], voice activity detection, speech segmentation, speaker embedding extraction, and speaker clustering. The multimodal approach fuses audio and video image input to produce more accurate results.

## Datasets Evaluation
The DER results from the audio-only diarization pipeline across various benchmarks and internal multi-speaker datasets.
| Test | DER(w/o Overlap-detection) | DER(w/ Overlap-detection)| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | [DiariZen_WavLM](https://github.com/BUTSpeechFIT/DiariZen) | 
|:-----:|:------:|:------:|:------:|:------:|
|[Aishell-4](https://arxiv.org/abs/2104.03603)|23.04%|**10.30%**|12.2%|11.7%|
|[Alimeeting](https://www.openslr.org/119/)|32.79%|19.73%|24.4%|**17.6%**|
|[AMI_SDM](https://groups.inf.ed.ac.uk/ami/corpus/)|35.76%|21.76%|22.4%|**15.4%**|
|[VoxConverse](https://github.com/joonson/voxconverse)|12.09%|11.75%|**11.3%**|28.39%|
|Meeting-CN_ZH-1|**16.80%**|18.91%|22.37%|32.66%|
|Meeting-CN_ZH-2|**11.98%**|12.78%|17.86%|18%|

The comparison of computational efficiency for audio-only diarization on the CPU device.
| | RTF(w/ Overlap-detection) | [pyannote.audio](https://github.com/pyannote/pyannote-audio) | [DiariZen_WavLM](https://github.com/BUTSpeechFIT/DiariZen) | 
|:-----:|:------:|:------:|:------:|
|RTF|**0.03**|0.19|0.3|

The DER results of two diarization pipelines on a multi-person conversation video dataset.
| Pipeline | DER |
|:-----:|:------:|
|Audio-only diarization|5.3%|
|Multimodal diarization|3.7%|

## Usage
### Quick Start
To use this diarization tool, follow the steps below:
1. Install required packages:
``` sh
pip install -r requirements.txt
```
2. For audio-only diarization, run:
``` sh
bash run_audio.sh
# Use the funasr model to transcribe into Chinese text.
bash run_audio.sh --stop_stage 8
```
3. For multimodal diarization, ensure that ffmpeg is available in your environment.
``` sh
sudo apt-get update
sudo apt-get install ffmpeg
bash run_video.sh
```
### Diarization with Overlap Detection
If you want to include overlapping speakers in final diarization results, you can run 
``` sh
bash run_audio.sh --include_overlap=true --hf_access_token=hf_xxx
```
The [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) is used as a overlapping speech detection module. Make sure to accept [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) user conditions and create an access token at [hf.co/settings/tokens](https://hf.co/settings/tokens)

### More Quick Audio Diarization
For those who wish to bypass the detailed stages in `run_audio.sh` and quickly obtain audio diarization results, run:
``` sh
# audio-only diarization
python speakerlab/bin/infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir]
# enable overlap detection
python speakerlab/bin/infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token]
# for more configurable parameters, you can refer to speakerlab/bin/infer_diarization.py
```
### Integration with Python Scripts
You can also integrate the diarization pipeline into your own Python scripts as follows:
```python
from speakerlab.bin.infer_diarization import Diarization3Dspeaker
wav_path = "audio.wav"
pipeline = Diarization3Dspeaker()
print(pipeline(wav_path, wav_fs=None, speaker_num=None)) # can also accept WAV data as input
```

## Limitations
- It may not perform well when the audio duration is too short (less than 30 seconds) and when the number of speakers is too large.
- The final accuracy is highly dependent on the performance of each modules. Therefore, using pretrained models that are more aligned with the test scenario may result in higher accuracy.
