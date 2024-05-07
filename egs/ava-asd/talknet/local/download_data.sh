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

stage=1
stop_stage=1
download_dir=$data/download_data

. utils/parse_options.sh || exit 1

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

# Download csv files
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "$(basename $0) Stage1: Prepare CSV files..."
  if [ ! -d $download_dir/csv ];then
    if [ ! -f $download_dir/csv.tar.gz ];then
      download_link=1C1cGxPHaJAl1NQ2i7IhRgWmdvsPhBCUy
      gdown $download_link -O $download_dir/csv.tar.gz
    fi
    tar -xzvf $download_dir/csv.tar.gz -C $download_dir/
  fi
fi

# Download data files
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(basename $0) Stage2: Download raw videos..."
  for split in trainval test;do
    mkdir -p $download_dir/orig_videos/$split
    cat $download_dir/csv/${split}_file_list.txt | while read video_name;do
      wget -P $download_dir/orig_videos/$split https://s3.amazonaws.com/ava-dataset/$split/$video_name
    done
  done
fi

# Extract audio data
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$(basename $0) Stage3: Extract raw audios..."
  for split in trainval test;do
    mkdir -p $download_dir/orig_audios/$split
    cat $download_dir/csv/${split}_file_list.txt | while read video_name;do
      input_video=$download_dir/orig_videos/$split/$video_name
      output_wav=$download_dir/orig_audios/$split/${video_name%.*}.wav
      ffmpeg -nostdin -y -i $input_video -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 16 $output_wav -loglevel panic
    done
  done
fi

# Extract audio clip data for training
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "$(basename $0) Stage4: Extract audio clips for training..."
  python local/extract_audio_clips.py --csv_ori $download_dir/csv/train_orig.csv --audio_ori_dir $download_dir/orig_audios/trainval --audio_out_dir $download_dir/clips_audios/train
  python local/extract_audio_clips.py --csv_ori $download_dir/csv/val_orig.csv --audio_ori_dir $download_dir/orig_audios/trainval --audio_out_dir $download_dir/clips_audios/val
  python local/extract_audio_clips.py --csv_ori $download_dir/csv/test_orig.csv --audio_ori_dir $download_dir/orig_audios/test --audio_out_dir $download_dir/clips_audios/test
fi

# Extract video clip data for training
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "$(basename $0) Stage5: Extract video clips for training..."
  echo "$(basename $0) Stage5: This stage is quite time-intensive, possibly requiring between 10 to 20 hours to complete."
  python local/extract_video_clips.py --csv_ori $download_dir/csv/train_orig.csv --video_ori_dir $download_dir/orig_videos/trainval \
  --video_out_dir $download_dir/clips_videos/train > /dev/null 2>&1 || { echo "Error: Failed to extract video clips of $download_dir/csv/train_orig.csv."; exit 1; }

  python local/extract_video_clips.py --csv_ori $download_dir/csv/val_orig.csv --video_ori_dir $download_dir/orig_videos/trainval \
  --video_out_dir $download_dir/clips_videos/val > /dev/null 2>&1 || { echo "Error: Failed to extract video clips of $download_dir/csv/val_orig.csv."; exit 1; }

  python local/extract_video_clips.py --csv_ori $download_dir/csv/test_orig.csv --video_ori_dir $download_dir/orig_videos/test \
  --video_out_dir $download_dir/clips_videos/test > /dev/null 2>&1 || { echo "Error: Failed to extract video clips of $download_dir/csv/test_orig.csv."; exit 1; }
fi
