#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=1
stop_stage=4

wav_list=examples/wav.list
exp=exp
conf_file=conf/diar.yaml
gpus="0 1 2 3"
nj=8

. local/parse_options.sh || exit 1

json_dir=$exp/json
embs_dir=$exp/embs
rttm_dir=$exp/rttm


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage1: Do vad for input wavs..."
  python local/voice_activity_detection.py --wavs $wav_list --out_file $json_dir/vad.json
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage2: Prepare subsegments info..."
  python local/prepare_subseg_json.py --vad $json_dir/vad.json --out_file $json_dir/subseg.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage3: Extract speaker embeddings..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  speaker_model_id=damo/speech_campplus_sv_zh-cn_16k-common
  torchrun --nproc_per_node=$nj local/extract_diar_embeddings.py --model_id $speaker_model_id --conf $conf_file \
          --subseg_json $json_dir/subseg.json --embs_out $embs_dir --gpu $gpus --use_gpu
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage4: Perform clustering and output sys rttms..."
  torchrun --nproc_per_node=$nj local/cluster_and_postprocess.py --conf $conf_file --embs_dir $embs_dir --rttm_dir $rttm_dir
fi
