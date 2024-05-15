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
  # In this stage we prepare the raw datasets
  echo "Stage 1: download aishell-4 and alimeeting datasets..."
  ./local/download_aishell_4_data.sh $work/corpus/
  ./local/download_alimeeting_data.sh $work/corpus/
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Prepare
  echo "Stage 2: prepare aishell-4 and alimeeting train and test files"
  python local/prepare_files_for_aishell_4.py
  python local/prepare_files_for_alimeeting.py

  # Merge files
  python local/merge_json_files_for_semantic_speaker.py
  python local/merge_json_files_for_semantic_speaker.py
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3]; then
  # Run speaker-turn detection
  echo "Stage 3: train speaker-turn detection model"
  python bin/run_dialogue_detection.py
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Run speaker-turn detection
  echo "Stage 3: test speaker-turn detection model"
  python bin/run_speaker_turn_detection.py
fi
