#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=1
stop_stage=3

data=data
exp=exp
exp_name=talknet
gpus="0 1 2 3"

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we prepare the raw datasets.
  echo "Stage1: Preparing AVA-ActiveSpeaker dataset..."
  ./local/download_data.sh  --stage 1 --stop_stage 5 --download_dir $data
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Train the talknet model.
  echo "Stage2: Training the talknet model..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --nproc_per_node=$num_gpu speakerlab/bin/train_asd.py --config conf/config.yaml --gpu $gpus \
           --train_csv $data/csv/train_loader.csv --val_csv $data/csv/val_loader.csv \
           --train_video_dir $data/clips_videos/train --train_audio_dir $data/clips_audios/train \
           --val_video_dir $data/clips_videos/val --val_audio_dir $data/clips_audios/val --exp_dir $exp_dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Evaluate the talknet model.
  echo "Stage3: Evaluating the talknet model..."
  nj=16
  torchrun --nproc_per_node=$nj speakerlab/bin/train_asd.py --config conf/config.yaml --gpu $gpus \
           --val_csv $data/csv/val_loader.csv --val_video_dir $data/clips_videos/val \
           --val_audio_dir $data/clips_audios/val --exp_dir $exp_dir --test
fi
