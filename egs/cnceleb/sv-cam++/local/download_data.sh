#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

. tools/parse_options.sh || exit 1

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

if [ ! -f ${download_dir}/cn-celeb_v2.tar.gz ]; then
  echo "Downloading cn-celeb_v2.tar.gz ..."
  wget --no-check-certificate https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz -P ${download_dir}
  md5=$(md5sum ${download_dir}/cn-celeb_v2.tar.gz | awk '{print $1}')
  [ $md5 != "7ab1b214028a7439e26608b2d5a0336c" ] && echo "Wrong md5sum of cn-celeb_v2.tar.gz" && exit 1
fi

if [ ! -f ${download_dir}/cn-celeb2_v2.tar.gz ]; then
  echo "Downloading cn-celeb2_v2.tar.gz ..."
  for part in a b c; do
    wget --no-check-certificate https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gza${part} -P ${download_dir} &
  done
  wait
  cat ${download_dir}/cn-celeb2_v2.tar.gza* >${download_dir}/cn-celeb2_v2.tar.gz
  md5=$(md5sum ${download_dir}/cn-celeb2_v2.tar.gz | awk '{print $1}')
  [ $md5 != "55c47cf0b6d0bf793e88bf79d5dfc660" ] && echo "Wrong md5sum of cn-celeb2_v2.tar.gz" && exit 1
fi

echo "Download success !!!"
