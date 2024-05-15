# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import yaml
import codecs
import json


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_data_csv(fpath):
    with open(fpath, newline="") as f:
        result = {}
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if 'ID' not in row:
                raise KeyError(
                    "CSV file has to have an 'ID' field, with unique ids for all data points."
                )

            data_id = row["ID"]
            del row["ID"]

            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            result[data_id] = row
    return result


def load_data_list(fpath):
    with open(fpath) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {idx: row for idx, row in enumerate(rows)}
    return result


def load_wav_scp(fpath):
    with open(fpath) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {i.split()[0]: i.split()[1] for i in rows}
    return result


def load_json_file(json_file):
    with codecs.open(json_file, "r", encoding="utf-8") as fr:
        data_dict = json.load(fr)
    return data_dict


def load_trans7time_list(filename):
    trans7time_list = []
    with open(filename, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            ps = line.strip().split()
            trans7time_list.append((
                ps[0], float(ps[1]), float(ps[2]), "".join(ps[3:])
            ))
    return trans7time_list


def write_json_file(json_file, data):
    assert str(json_file).endswith(".json") or str(json_file).endswith(".JSON")
    with codecs.open(json_file, "w", encoding="utf-8") as fw:
        json.dump(data, fw, indent=2, ensure_ascii=False)


def write_wav_scp(fpath, wav_scp):
    with open(fpath, "w") as f:
        for key, value in wav_scp.items():
            f.write(f"{key} {value}\n")


def write_trans7time_list(fpath, trans7time_list):
    """
        trans7time_list: [(spk_id, start_time, end_time, text)]
    """
    with open(fpath, 'w') as fw:
        for spk_id, start_time, end_time, text in trans7time_list:
            fw.write(f'{spk_id} {start_time} {end_time} {text}\n')
