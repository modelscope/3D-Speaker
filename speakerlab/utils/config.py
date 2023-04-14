# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import yaml

class Config(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            self.__dict__[key] = value


def convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--") :] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()


def yaml_config_loader(conf_file, overrides=None):
    with open(conf_file, "r") as fr:
        conf_dict = yaml.load(fr, Loader=yaml.FullLoader)
    if overrides is not None:
        overrides = yaml.load(overrides, Loader=yaml.FullLoader)
        conf_dict.update(overrides)
    return conf_dict


def build_config(config_file, overrides=None, copy=False):
    if config_file.endswith(".yaml"):
        if overrides is not None:
            overrides = convert_to_yaml(overrides)
        conf_dict = yaml_config_loader(config_file, overrides)
        if copy and 'exp_dir' in conf_dict:
            os.makedirs(conf_dict['exp_dir'], exist_ok=True)
            saved_path = os.path.join(conf_dict['exp_dir'], 'config.yaml')
            with open(saved_path, 'w') as f:
                f.write(yaml.dump(conf_dict))
    else:
        raise ValueError("Unknown config file format")

    return Config(conf_dict)
