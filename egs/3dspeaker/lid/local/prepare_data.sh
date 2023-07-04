#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Hui Wang (tongmu.wh@alibaba-inc.com)
#
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

stage=-1
stop_stage=-1
data=data

. utils/parse_options.sh || exit 1

download_dir=${data}/download_data
rawdata_dir=${data}/raw_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip, train.tar.gz test.tar.gz 3dspeaker_files.tar.gz"
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip train.tar.gz test.tar.gz 3dspeaker_files.tar.gz; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/3dspeaker ]; then
    mkdir -p ${rawdata_dir}/3dspeaker
    mkdir -p ${rawdata_dir}/3dspeaker/test ${rawdata_dir}/3dspeaker/train ${rawdata_dir}/3dspeaker/files
    tar -zxvf ${download_dir}/train.tar.gz -C ${rawdata_dir}/3dspeaker/
    tar -xzvf ${download_dir}/test.tar.gz -C ${rawdata_dir}/3dspeaker/
    tar -xzvf ${download_dir}/3dspeaker_files.tar.gz -C ${rawdata_dir}/3dspeaker/
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for 3dspeaker datasets"
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/musan ${data}/rirs ${data}/3dspeaker
  # musan
  find $(pwd)/${rawdata_dir}/musan/noise/free-sound -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # rirs
  awk '{print $5}' $(pwd)/${rawdata_dir}/RIRS_NOISES/real_rirs_isotropic_noises/rir_list | xargs -I {} echo {} $(pwd)/${rawdata_dir}/{} > ${data}/rirs/wav.scp
  # 3dspeaker
  base_path=${data}/3dspeaker/
  ## train
  train_base_path=${base_path}/train
  train_rawdata_path=${rawdata_dir}/3dspeaker/
  mkdir -p $train_base_path
  awk -v base_path="${train_rawdata_path}" '{print $1" "base_path $2}' ${rawdata_dir}/3dspeaker/files/lid_train_wav.scp > ${train_base_path}/wav.scp
  perl -e '
    open(F, "<$ARGV[0]") || die "Could not open file $_";
    while(<F>) {
      @A = split;
      @A>=1 || die "Invalid file line $_";
      $seen{$A[0]} = 1;
    }
    open(F, "<$ARGV[1]") || die "Could not open file $_";
    while(<F>) {
      @B = split /,/;
      @B>=1 || die "Invalid id file file line $_";
      if ($seen{$B[0]}){
        $B[5] =~ s/ /_/g;
        print "$B[0] $B[5]";
      }
    }
  ' ${train_base_path}/wav.scp ${rawdata_dir}/3dspeaker/files/train_utt2info.csv > ${train_base_path}/utt2spk

  ## test
  test_base_path=${base_path}/test
  test_rawdata_path=${rawdata_dir}/3dspeaker/
  mkdir -p $test_base_path
  awk -v base_path="${test_rawdata_path}" '{print $1" "base_path $2}' ${rawdata_dir}/3dspeaker/files/lid_test_wav.scp > ${test_base_path}/wav.scp
  perl -e '
    open(F, "<$ARGV[0]") || die "Could not open file $_";
    while(<F>) {
      @A = split;
      @A>=1 || die "Invalid file line $_";
      $seen{$A[0]} = 1;
    }
    open(F, "<$ARGV[1]") || die "Could not open file $_";
    while(<F>) {
      @B = split /,/;
      @B>=1 || die "Invalid id file file line $_";
      if ($seen{$B[0]}){
        $B[5] =~ s/ /_/g;
        print "$B[0] $B[5]";
      }
    }
  ' ${test_base_path}/wav.scp ${rawdata_dir}/3dspeaker/files/test_utt2info.csv > ${test_base_path}/utt2spk
  
  echo "Data Preparation Success !!!"
fi
