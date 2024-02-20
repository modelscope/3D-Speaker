#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Yafeng Chen (chenyafeng.cyf@alibaba-inc.com)
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
  echo "Download musan.tar.gz, rirs_noises.zip, cn-celeb_v2.tar.gz cn-celeb2_v2.tar.gz."
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip cn-celeb_v2.tar.gz cn-celeb2_v2.tar.gz; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/CN-Celeb_flac ]; then
    tar -xzvf ${download_dir}/cn-celeb_v2.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/CN-Celeb2_flac ]; then
    tar -xzvf ${download_dir}/cn-celeb2_v2.tar.gz -C ${rawdata_dir}
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "convert flac to wav ..."
  python local/flac2wav.py \
      --dataset_dir ${rawdata_dir}/CN-Celeb_flac \
      --nj 16

  python local/flac2wav.py \
      --dataset_dir ${rawdata_dir}/CN-Celeb2_flac \
      --nj 16
  echo "convert success"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/musan ${data}/rirs ${data}/cnceleb_train ${data}/eval
  # musan
  find $(pwd)/${rawdata_dir}/musan/noise/free-sound -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' > ${data}/musan/wav.scp
  # rirs
  awk '{print $5}' $(pwd)/${rawdata_dir}/RIRS_NOISES/real_rirs_isotropic_noises/rir_list | xargs -I {} echo {} $(pwd)/${rawdata_dir}/{} > ${data}/rirs/wav.scp

  echo "Prepare train data including CN-Celeb_wav/dev and CN-Celeb2_wav ..."
  [ -f ${data}/cnceleb_train/wav.scp ] && rm ${data}/cnceleb_train/wav.scp
  for spk in `cat ${rawdata_dir}/CN-Celeb_flac/dev/dev.lst`; do
    find ${rawdata_dir}/CN-Celeb_wav/data/${spk} -name "*.wav" | \
      awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >> ${data}/cnceleb_train/wav.scp
  done

  for spk in `cat ${rawdata_dir}/CN-Celeb2_flac/spk.lst`; do
    find ${rawdata_dir}/CN-Celeb2_wav/data/${spk} -name "*.wav" | \
      awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >> ${data}/cnceleb_train/wav.scp
  done

  awk '{print $1}' ${data}/cnceleb_train/wav.scp | awk -F "/" '{print $0,$1}' > ${data}/cnceleb_train/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${data}/cnceleb_train/utt2spk >${data}/cnceleb_train/spk2utt

  echo "Prepare data for testing ..."
  find ${rawdata_dir}/CN-Celeb_wav/eval -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort > ${data}/eval/wav.scp
  awk '{print $1}' ${data}/eval/wav.scp | awk -F "[/-]" '{print $0,$2}' > ${data}/eval/utt2spk

  echo "Prepare data for enroll ..."
  awk '{print $0}' ${rawdata_dir}/CN-Celeb_flac/eval/lists/enroll.map | \
    awk -v p=${rawdata_dir}/CN-Celeb_wav/data '{for(i=2;i<=NF;i++){print $i, p"/"$i}}' > ${data}/eval/enroll.scp
  cat ${data}/eval/enroll.scp >> ${data}/eval/wav.scp
  awk '{print $1}' ${data}/eval/enroll.scp | awk -F "/" '{print $0,$1"-enroll"}' >> ${data}/eval/utt2spk
  cp ${rawdata_dir}/CN-Celeb_flac/eval/lists/enroll.map ${data}/eval/enroll.map

  echo "Prepare evalution trials ..."
  mkdir -p ${data}/cnceleb_test
  awk '{if($3==0)label="nontarget";else{label="target"}; print "enroll/" $1 ".wav", $2, label}' ${rawdata_dir}/CN-Celeb_flac/eval/lists/trials.lst > ${data}/cnceleb_test/trials
  cp ${data}/eval/wav.scp ${data}/cnceleb_test/

  echo "Success !!! Now data preparation is done !!!"
fi
