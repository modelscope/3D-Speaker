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
  echo "Stage 2: prepare aishell-4 and alimeeting train and test files"
  aishell_4_file_path=$work/corpus/aishell_4/semantic_files/
  alimeeting_file_path=$work/corpus/alimeeting/semantic_files/
  python local/prepare_files_for_aishell_4.py --home_path $work/corpus/aishell_4/ --save_path $aishell_4_file_path
  python local/prepare_files_for_alimeeting.py --home_path $work/corpus/alimeeting/ --save_path $alimeeting_file_path

  # Merge files, and prepare train, valid, test
  cat $aishell_4_file_path/train_L_trans7time.scp $aishell_4_file_path/train_M_trans7time.scp $aishell_4_file_path/train_S_trans7time.scp $alimeeting_file_path/train_far_trans7time.scp > $work/corpus/total_train_trans7time.scp
  cat $alimeeting_file_path/eval_far_trans7time.scp > $work/corpus/total_valid_trans7time.scp
  cat $aishell_4_file_path/test_trans7time.scp $alimeeting_file_path/test_far_trans7time.scp > $work/corpus/total_test_trans7time.scp

  # Prepare files in json format for model input
  json_path=$work/corpus/json_files/
  python local/prepare_json_files_for_semantic_speaker.py --flag train --trans7time_scp_file $work/corpus/total_train_trans7time.scp --save_path $json_path --sentence_length 96 --sentence_shift 32
  python local/prepare_json_files_for_semantic_speaker.py --flag valid --trans7time_scp_file $work/corpus/total_train_trans7time.scp --save_path $json_path --sentence_length 96 --sentence_shift 32
  python local/prepare_json_files_for_semantic_speaker.py --flag test --trans7time_scp_file $work/corpus/total_train_trans7time.scp --save_path $json_path --sentence_length 96 --sentence_shift 32
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3]; then
  # Run dialogue detection
  echo "Stage 3: train dialogue detection model"
  python bin/run_dialogue_detection.py
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Run dialogue detection
  echo "Stage 3: test dialogue detection model"
  python bin/run_dialogue_detection.py
fi
