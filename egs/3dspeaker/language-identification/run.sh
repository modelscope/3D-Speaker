#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# The scripts in the sv-cam++ and sv-eres2net recipe are re-used for language identification after slightly modified.
# The utt2spk in this recipe is equivalent to utt2lang


set -e
. ./path.sh || exit 1

stage=1
stop_stage=5


data=data
exp=exp
# cam++ or eres2net
exp_name=eres2net
# exp_name=cam++
gpus="0 1 2 3"

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we prepare the raw datasets.
  echo "Stage1: Preparing 3D-Speaker dataset..."
  ./local/prepare_data.sh --stage 1 --stop_stage 3 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # In this stage we prepare the data index files for training.
  echo "Stage2: Preparing training data index files..."
  python local/prepare_data_csv.py --data_dir $data/3dspeaker/train
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Train the model.
  echo "Stage3: Training the model..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  # Using CAM++ model for language identification
  # torchrun --nproc_per_node=$num_gpu speakerlab/bin/train.py --config conf/cam++.yaml --gpu $gpus \
  #          --data $data/3dspeaker/train/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
  # Using ERes2Net model for language identification
  torchrun --nproc_per_node=$num_gpu speakerlab/bin/train.py --config conf/eres2net.yaml --gpu $gpus \
           --data $data/3dspeaker/train/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Output the prediction results.
  echo "Stage4: Predicting the test data..."
  nj=8
  torchrun --nproc_per_node=$nj local/predict.py --exp_dir $exp_dir \
           --data $data/3dspeaker/test/wav.scp --use_gpu --gpu $gpus
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Output score metrics.
  echo "Stage5: Computing score metrics..."
  cat $exp_dir/results/predicts/predict*.txt > $exp_dir/results/predict.txt
  python local/compute_acc.py --predict $exp_dir/results/predict.txt --ground_truth $data/3dspeaker/test/utt2spk \
                  --out_dir $exp_dir/results
fi
