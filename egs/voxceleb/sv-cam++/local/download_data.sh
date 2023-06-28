#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

download_dir=data/download_data

. utils/parse_options.sh || exit 1

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

if [ ! -f ${download_dir}/musan.tar.gz ]; then
  echo "Downloading musan.tar.gz ..."
  wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${download_dir}
  md5=$(md5sum ${download_dir}/musan.tar.gz | awk '{print $1}')
  [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1
fi

if [ ! -f ${download_dir}/rirs_noises.zip ]; then
  echo "Downloading rirs_noises.zip ..."
  wget --no-check-certificate https://us.openslr.org/resources/28/rirs_noises.zip -P ${download_dir}
  md5=$(md5sum ${download_dir}/rirs_noises.zip | awk '{print $1}')
  [ $md5 != "e6f48e257286e05de56413b4779d8ffb" ] && echo "Wrong md5sum of rirs_noises.zip" && exit 1
fi

if [ ! -f ${download_dir}/vox1_test_wav.zip ]; then
  echo "Downloading vox1_test_wav.zip ..."
  wget --no-check-certificate https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip -P ${download_dir}
  md5=$(md5sum ${download_dir}/vox1_test_wav.zip | awk '{print $1}')
  [ $md5 != "185fdc63c3c739954633d50379a3d102" ] && echo "Wrong md5sum of vox1_test_wav.zip" && exit 1
fi

if [ ! -f ${download_dir}/vox1_dev_wav.zip ]; then
  echo "Downloading vox1_dev_wav.zip ..."
  for part in a b c d; do
    wget --no-check-certificate https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta${part} -P ${download_dir} &
  done
  wait
  cat ${download_dir}/vox1_dev* >${download_dir}/vox1_dev_wav.zip
  md5=$(md5sum ${download_dir}/vox1_dev_wav.zip | awk '{print $1}')
  [ $md5 != "ae63e55b951748cc486645f532ba230b" ] && echo "Wrong md5sum of vox1_dev_wav.zip" && exit 1
fi

if [ ! -f ${download_dir}/vox2_aac.zip ]; then
  echo "Downloading vox2_aac.zip ..."
  for part in a b c d e f g h; do
    wget --no-check-certificate https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_parta${part} -P ${download_dir} &
  done
  wait
  cat ${download_dir}/vox2_dev_aac* >${download_dir}/vox2_aac.zip
  md5=$(md5sum ${download_dir}/vox2_aac.zip | awk '{print $1}')
  [ $md5 != "bbc063c46078a602ca71605645c2a402" ] && echo "Wrong md5sum of vox2_aac.zip" && exit 1
fi

echo "Download success !!!"
