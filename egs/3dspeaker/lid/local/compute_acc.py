# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
import numpy as np
import pandas as pd
from speakerlab.utils.utils import get_utt2spk_dict

parser = argparse.ArgumentParser(description='Output the score.')
parser.add_argument('--predict', default='', type=str, help='Prediction results')
parser.add_argument('--ground_truth', default='', type=str, help='Ground truth')
parser.add_argument('--out_dir', default='', type=str, help='Out dir')

def main():
    args = parser.parse_args()
    predict = get_utt2spk_dict(args.predict)
    ground = get_utt2spk_dict(args.ground_truth)
    langs = list(set(ground.values()))
    langs.sort()
    
    confusion = np.zeros((len(langs), len(langs)))
    sum = 0
    cor = 0
    for utt_id in predict:
        sum += 1
        confusion[langs.index(ground[utt_id])][langs.index(predict[utt_id])] += 1
        if predict[utt_id] == ground[utt_id]:
            cor += 1

    os.makedirs(args.out_dir, exist_ok=True)
    with open('%s/acc.res'%args.out_dir, 'w') as f:
        f.write("Acc:%.2f%%"%(cor/sum*100))
    print("Acc:%.2f%%"%(cor/sum*100))

    writeExcel(confusion, langs, '%s/predict.xlsx'%args.out_dir)

def writeExcel(data, langs, excel_path):
        assert data.shape[0] == len(langs)
        num = data.shape[0]
        form = np.zeros((num + 1, num + 4))
        for i in range(num):
            form[num, i] = sum(data[:, i])
        for i in range(num):
            form[i, num] = sum(data[i])
            form[i, num + 1] = data[i, i] / (float(form[i, num]) + 1e-10)
            form[i, num + 2] = data[i, i] / (float(form[num, i]) + 1e-10)
            form[i, num + 3] = 2 * form[i, num + 1] * form[i, num + 2] / (
                    float((form[i, num + 1] + form[i, num + 2])) + 1e-10)
        form[:num, :num] = data
        data_pd = pd.DataFrame(form)
        data_pd.columns = langs + ['Total', 'Recall', 'Precision', 'F1']
        data_pd.index = langs + ['Total']
        writer = pd.ExcelWriter(excel_path)
        data_pd.to_excel(writer, float_format='%.3f')
        writer.close()

if __name__ == "__main__":
    main()