#!/bin/bash

# Copyright (c) 2023 Yafeng Chen (chenyafeng.cyf@alibaba-inc.com)
#               2023 Luyao Cheng (shuli.cly@alibaba-inc.com)
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

if [ ! -f ${download_dir}/test.tar.gz ]; then
    echo "Downloading 3dspeaker test.tar.gz"
    wget --no-check-certificate https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/test.tar.gz -P ${download_dir}
    md5=$(md5sum ${download_dir}/test.tar.gz | awk '{print $1}')
    [ $md5 != "45972606dd10d3f7c1c31f27acdfbed7" ] && echo "Wrong md5sum of 3dspeaker test.tar.gz" && exit 1
fi

if [ ! -f ${download_dir}/train.tar.gz ]; then
    echo "Downloading 3dspeaker train.tar.gz"
    # md5 value of part files:
    # train.tar.gz-part-a 4109addde41d88760947263f18117ac3 
    # train.tar.gz-part-b 5a17ef2fa28b1b9e340277edffb8b51c 
    # train.tar.gz-part-c bd2ce08f5b51005b66afe484b01a4a59 
    # train.tar.gz-part-d 5cd31d961d2d5211aea38b8b95f7239a 
    # train.tar.gz-part-e 58f3fb7d28ae7f4b65ee35a1ed7ab106 
    # train.tar.gz-part-f be64551c030e8087562a10df2c74ccb1 
    for part in a b c d e f; do
        wget --no-check-certificate https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/train.tar.gz-part-${part} -P ${download_dir}
    done
    wait
    cat ${download_dir}/train.tar.gz-part-* > ${download_dir}/train.tar.gz
    md5=$(md5sum ${download_dir}/train.tar.gz | awk '{print $1}')
    [ $md5 != "c2cea55fd22a2b867d295fb35a2d3340" ] && echo "Wrong md5sum of 3dspeaker train.tar.gz" && exit 1
fi

if [ ! -f ${download_dir}/3dspeaker_files.tar.gz ]; then
    echo "Downloading 3dspeaker utterances files"
    wget --no-check-certificate https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/3dspeaker_files.tar.gz -P ${download_dir}
fi

echo "Download success !!!"
