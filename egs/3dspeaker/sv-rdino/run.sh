#!/bin/bash


set -e
. ./path.sh || exit 1

stage=1
stop_stage=4

data=data
exp=exp
exp_name=rdino
gpus="0 1 2 3"

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage1: Preparing 3D Speaker dataset ..."
  ./local/prepare_data_rdino.sh --stage 1 --stop_stage 3 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage2: Training the speaker model ..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --nproc_per_node=$num_gpu speakerlab/bin/train_rdino.py --config conf/rdino.yaml --gpu $gpus \
           --data $data/3dspeaker/train//wav.scp --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage4: Extracting speaker embeddings ..."
  torchrun --nproc_per_node=8 speakerlab/bin/extract_ssl.py --exp_dir $exp_dir \
           --data $data/3dspeaker/test/wav.scp --use_gpu --gpu $gpus
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage5: Computing score metrics ..."
  trials="$data/3dspeaker/trials/trials_cross_device $data/3dspeaker/trials/trials_cross_distance $data/3dspeaker/trials/trials_cross_dialect"
  python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_dir/embeddings --test_data $exp_dir/embeddings \
                                                 --scores_dir $exp_dir/scores --trials $trials --p_target 0.05
fi
