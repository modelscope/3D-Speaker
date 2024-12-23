#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script performs speaker diarization task based on video input.
# It extracts both visual and audio speaker embeddings and generates more accurate results than audio-only diarization.

set -e
. ./path.sh || exit 1

stage=1
stop_stage=6

examples=examples
exp=exp_video
conf_file=conf/diar_video.yaml
onnx_dir=pretrained_models
gpus="0 1 2 3"
nj=4

. local/parse_options.sh || exit 1

video_list=$examples/video.list
raw_data_dir=$exp/raw
visual_embs_dir=$exp/embs_video
rttm_dir=$exp/rttm

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  if [ ! -f "$video_list" ]; then
    echo "$(basename $0) Stage1: Prepare input videos..."
    mkdir -p $examples
    wget "https://modelscope.cn/api/v1/models/iic/speech_campplus_speaker-diarization_common/\
resolve/master/examples/7speakers_example.mp4" -O $examples/7speakers_example.mp4
    wget "https://modelscope.cn/api/v1/models/iic/speech_campplus_speaker-diarization_common/\
resolve/master/examples/7speakers_example.rttm" -O $examples/7speakers_example.rttm
    echo "examples/7speakers_example.mp4" > $examples/video.list
    echo "examples/7speakers_example.rttm" > $examples/refrttm.list
  else
    echo "$(basename $0) Stage 1: $video_list exists. Skip this stage."
  fi
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo "$(basename $0) Stage2: Prepare onnx files and extrack raw videos and audios..."
  mkdir -p $onnx_dir
  mkdir -p $raw_data_dir
  for m in version-RFB-320.onnx asd.onnx fqa.onnx face_recog_ir101.onnx; do
    if [ ! -e $onnx_dir/$m ]; then
      echo "$(basename $0) Stage2: Download pretrained models $m"
      wget -O $onnx_dir/$m "https://modelscope.cn/api/v1/models/iic/speech_campplus_speaker-diarization_common/resolve/master/onnx/$m"
    fi
  done
  cat $video_list | while read video_file; do
    filename=$(basename $video_file)
    out_video_file=$raw_data_dir/${filename%.*}.mp4
    out_wav_file=$raw_data_dir/${filename%.*}.wav
    if [ ! -e $out_video_file ]; then
      echo "$(basename $0) Stage2: Extract video from $filename"
      ffmpeg -nostdin -y -i $video_file -qscale:v 2 -threads 16 -async 1 -r 25 $out_video_file -loglevel panic
    fi
    if [ ! -e $out_wav_file ]; then
      echo "$(basename $0) Stage2: Extract audio from $filename"
      ffmpeg -nostdin -y -i $out_video_file -qscale:a 0 -ac 1 -vn -threads 16 -ar 16000 $out_wav_file -loglevel panic
    fi
  done
fi

# write the input pair data list
cat $video_list | while read video_file; do filename=$(basename $video_file);echo $raw_data_dir/${filename%.*}.mp4;done > $raw_data_dir/video.list
cat $video_list | while read video_file; do filename=$(basename $video_file);echo $raw_data_dir/${filename%.*}.wav;done > $raw_data_dir/wav.list

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$(basename $0) Stage3: Extract audio speaker embeddings..."
  bash run_audio.sh --stage 2 --stop_stage 4 --examples $raw_data_dir --exp $exp
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "$(basename $0) Stage4: Extract visual speaker embeddings..."
  torchrun --nproc_per_node=$nj local/extract_visual_embeddings.py --conf $conf_file --videos $raw_data_dir/video.list \
          --vad $exp/json/vad.json --onnx_dir $onnx_dir --embs_out $visual_embs_dir --gpu $gpus --use_gpu
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "$(basename $0) Stage5: Clustering for both type of speaker embeddings..."
  torchrun --nproc_per_node=$nj local/cluster_and_postprocess.py --conf $conf_file --wavs $raw_data_dir/wav.list \
          --audio_embs_dir $exp/embs --visual_embs_dir $visual_embs_dir --rttm_dir $rttm_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "$(basename $0) Stage6: Get the final metrics..."
  ref_rttm_list=$examples/refrttm.list
  if [ -f $ref_rttm_list ]; then
    cat $ref_rttm_list | while read line;do cat $line;done > $exp/concat_ref_rttm
    echo "Computing DER..."
    python local/compute_der.py --exp_dir $exp --ref_rttm $exp/concat_ref_rttm
  else
    echo "Refrttm.list is not detected. Can't calculate the result"
  fi
fi
