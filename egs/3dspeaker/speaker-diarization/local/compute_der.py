# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import argparse
import shutil
import numpy as np
from speakerlab.utils.utils import parse_config, get_logger
from DER import DER


def main(args):
    logger = get_logger()
    sys_rttm_dir = os.path.join(args.exp_dir, 'rttm')
    result_dir = os.path.join(args.exp_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    concate_rttm_file = sys_rttm_dir + "/sys_output_rttm"
    if os.path.exists(concate_rttm_file):
        os.remove(concate_rttm_file)

    meta_file = os.path.join(args.exp_dir, 'json/subseg.json')
    with open(meta_file, "r") as f:
        full_meta = json.load(f)

    all_keys = full_meta.keys()
    A = ['_'.join(word.rstrip().split("_")[:-2]) for word in all_keys]
    all_rec_ids = list(set(A))
    all_rec_ids.sort()
    if len(all_rec_ids) <= 0:
        msg = "[ERROE] No recording IDs found! Please check if %s file is properly generated."%meta_file
        print(msg)
        sys.exit()

    out_rttm_files = []
    for rec_id in all_rec_ids:
        out_rttm_files.append(os.path.join(sys_rttm_dir, rec_id+'.rttm'))

    logger.info("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in out_rttm_files:
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)
    
    if args.ref_rttm != '':
        [MS, FA, SER, DER_] = DER(
            args.ref_rttm,
            concate_rttm_file,
        )
        msg = ', '.join(['MS: %f' % MS,
                'FA: %f' % FA,
                'SER: %f' % SER,
                'DER: %f' % DER_])
        logger.info(msg)
        with open('%s/der.txt' %result_dir,'w') as f:
            f.write(msg)
    else: 
        msg = '[INFO] There is no ref rttm file provided. Computing DER is Failed.'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',
                        type=str,
                        default="",
                        help="exp dir")
    parser.add_argument('--ref_rttm',
                        type=str,
                        default="",
                        help="ref rttm file")
    args = parser.parse_args()
    main(args)