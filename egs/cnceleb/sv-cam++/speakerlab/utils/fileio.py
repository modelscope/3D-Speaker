# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import yaml

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
