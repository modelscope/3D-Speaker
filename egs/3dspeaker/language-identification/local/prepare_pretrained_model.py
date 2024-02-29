#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import sys
import wget
import os

parser = argparse.ArgumentParser(description='Preparing pretrained paraformer')
parser.add_argument('--pretrained_model_dir', default='pretrained', type=str, help='Local model dir')

def main():

    args, _ = parser.parse_known_args(sys.argv[1:])
    os.makedirs(args.pretrained_model_dir, exist_ok=True)
    url_model = "https://modelscope.cn/api/v1/models/iic/speech_eres2net_base_lre_en-cn_16k/repo" \
                "?Revision=master&FilePath=pretrained_paraformer/model.pb"
    url_cmvn = "https://modelscope.cn/api/v1/models/iic/speech_eres2net_base_lre_en-cn_16k/repo" \
            "?Revision=master&FilePath=pretrained_paraformer/am.mvn"
    url_config = "https://modelscope.cn/api/v1/models/iic/speech_eres2net_base_lre_en-cn_16k/repo" \
                "?Revision=master&FilePath=pretrained_paraformer/config.yaml"
    wget.download(url_model, args.pretrained_model_dir)
    wget.download(url_cmvn, args.pretrained_model_dir)
    wget.download(url_config, args.pretrained_model_dir)


if __name__ == '__main__':
    
    main()