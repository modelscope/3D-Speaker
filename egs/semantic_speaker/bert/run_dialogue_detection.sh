#!/bin/bash

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

. ./path.sh

stage=1
stop_stage=3

work=$1


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we download the raw datasets
  echo "Stage 1: download aishell-4 and alimeeting datasets..."
  mkdir -p $work/corpus/
  mkdir -p $work/corpus/aishell_4/
  bash local/download_aishell_4_data.sh $work/corpus/aishell_4/
  mkdir -p $work/corpus/alimeeting/
  bash local/download_alimeeting_data.sh $work/corpus/alimeeting/
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Prepare semantic files
  echo "Stage 2.1: prepare aishell-4 and alimeeting train and test files"
  aishell_4_file_path=$work/corpus/aishell_4/semantic_files/
  alimeeting_file_path=$work/corpus/alimeeting/semantic_files/
  python local/prepare_files_for_aishell_4.py --home_path $work/corpus/aishell_4/ --save_path $aishell_4_file_path
  python local/prepare_files_for_alimeeting.py --home_path $work/corpus/alimeeting/ --save_path $alimeeting_file_path

  # Merge files, and prepare train, valid, test
  cat $aishell_4_file_path/train_L_trans7time.scp $aishell_4_file_path/train_M_trans7time.scp $aishell_4_file_path/train_S_trans7time.scp $alimeeting_file_path/train_far_trans7time.scp > $work/corpus/total_train_trans7time.scp
  cat $alimeeting_file_path/eval_far_trans7time.scp > $work/corpus/total_valid_trans7time.scp
  cat $aishell_4_file_path/test_trans7time.scp $alimeeting_file_path/test_far_trans7time.scp > $work/corpus/total_test_trans7time.scp

  # Prepare files in json format for model input
  echo "Stage 2.2: prepare json files for semantic tasks"
  json_path=$work/corpus/json_files/
  mkdir -p $json_path
  python local/prepare_json_files_for_semantic_speaker.py \
    --flag train --trans7time_scp_file $work/corpus/total_train_trans7time.scp --save_path $json_path \
    --sentence_length 96 --sentence_shift 32
  python local/prepare_json_files_for_semantic_speaker.py \
    --flag valid --trans7time_scp_file $work/corpus/total_valid_trans7time.scp --save_path $json_path \
    --sentence_length 96 --sentence_shift 32
  python local/prepare_json_files_for_semantic_speaker.py \
    --flag test --trans7time_scp_file $work/corpus/total_test_trans7time.scp --save_path $json_path \
    --sentence_length 96 --sentence_shift 32
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Run dialogue detection
  echo "Stage 3: train and test dialogue detection model"
  json_path=$work/corpus/json_files/
  output_path=$work/dialogue_detection_experiments/
  mkdir -p $output_path
  CUDA_VISIBLE_DEVICES=0,1,2,3 python bin/run_dialogue_detection.py \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 128 --pad_to_max_length \
    --train_file $json_path/train.dialogue_detection.json \
    --validation_file $json_path/valid.dialogue_detection.json \
    --test_file $json_path/test.dialogue_detection.json \
    --do_train --do_eval --do_predict \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --num_train_epochs 5 \
    --output_dir $output_path --overwrite_output_dir
fi
