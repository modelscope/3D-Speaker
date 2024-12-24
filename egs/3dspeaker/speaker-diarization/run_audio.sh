#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script performs speaker diarization task based on audio-only input, 
# in contrast to "run_video.sh" which is based on video and audio input.

set -e
. ./path.sh || exit 1

stage=1
stop_stage=7

conf_file=conf/diar.yaml
gpus="0 1 2 3"
nj=4
include_overlap=false
hf_access_token=

examples=examples
exp=exp

. local/parse_options.sh || exit 1

wav_list=$examples/wav.list
json_dir=$exp/json
embs_dir=$exp/embs
rttm_dir=$exp/rttm

if [ "$include_overlap" = true ] && [ -z "$hf_access_token" ]; then
  echo "[ERROR]: The hf_access_token is empty. If \"include_overlap\" is set to true,\
    the \"hf_access_token\" for \"pyannote/segmentation-3.0\" should be provided." && exit 1
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  if [ ! -f "$wav_list" ]; then
    echo "$(basename $0) Stage 1: Prepare input wavs..."
    mkdir -p $examples
    wget "https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common/\
resolve/master/examples/2speakers_example.wav" -O $examples/2speakers_example.wav
    wget "https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common/\
resolve/master/examples/2speakers_example.rttm" -O $examples/2speakers_example.rttm
    echo "examples/2speakers_example.wav" > $examples/wav.list
    echo "examples/2speakers_example.rttm" > $examples/refrttm.list
  else
    echo "$(basename $0) Stage 1: $wav_list exists. Skip this stage."
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  if [ "$include_overlap" = true ]; then
    echo "$(basename $0) Stage2: Do overlap detection for input wavs..."
    python local/overlap_detection.py --wavs $wav_list --out_dir $json_dir --hf_access_token $hf_access_token
  fi
  echo "$(basename $0) Stage2: Do vad for input wavs..."
  python local/voice_activity_detection.py --wavs $wav_list --out_file $json_dir/vad.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$(basename $0) Stage3: Prepare subsegments info..."
  python local/prepare_subseg_json.py --vad $json_dir/vad.json --out_file $json_dir/subseg.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "$(basename $0) Stage4: Extract speaker embeddings..."
  # Set speaker_model_id to damo/speech_eres2net_sv_zh-cn_16k-common when using eres2net 
  speaker_model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
  torchrun --nproc_per_node=$nj local/extract_diar_embeddings.py --model_id $speaker_model_id --conf $conf_file \
          --subseg_json $json_dir/subseg.json --embs_out $embs_dir --gpu $gpus --use_gpu
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "$(basename $0) Stage5: Perform clustering and postprocessing, and output sys rttms..."
  if [ "$include_overlap" = true ]; then
    cluster_rttm_dir=$rttm_dir/intermediate
  else
    cluster_rttm_dir=$rttm_dir
  fi
  torchrun --nproc_per_node=$nj local/cluster_and_postprocess.py --conf $conf_file --wavs $wav_list \
          --audio_embs_dir $embs_dir --rttm_dir $cluster_rttm_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  if [ "$include_overlap" = true ]; then
    echo "$(basename $0) Stage6: Do overlap detection postprocess..."
    python local/refine_with_OD.py --init_rttm_dir $rttm_dir/intermediate --rttm_dir $rttm_dir \
        --segmentation_dir $json_dir --hf_access_token $hf_access_token
  fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "$(basename $0) Stage7: Get the DER metrics..."
  ref_rttm_list=$examples/refrttm.list
  if [ -f $ref_rttm_list ]; then
    cat $ref_rttm_list | while read line;do cat $line;done > $exp/concat_ref_rttm
    echo "Computing DER..."
    python local/compute_der.py --exp_dir $exp --ref_rttm $exp/concat_ref_rttm
  else
    echo "Refrttm.list is not detected. Can't calculate the result"
  fi
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "$(basename $0) Stage8: Generate segmented transcription results with speaker ID...This step may be time-consuming."
  torchrun --nproc_per_node=$nj local/out_transcription.py --exp_dir $exp --gpu $gpus
fi
