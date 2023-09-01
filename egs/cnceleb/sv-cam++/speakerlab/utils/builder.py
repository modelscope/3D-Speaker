# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import re
import importlib
from speakerlab.utils.config import Config


def dynamic_import(import_path):
    module_name, obj_name = import_path.rsplit('.', 1)
    m = importlib.import_module(module_name)
    return getattr(m, obj_name)

def is_ref_type(value: str):
    assert isinstance(value, str), 'Input value is not a str.'
    if re.match('^<[a-zA-Z]\w*>$', value):
        return True
    else:
        return False

def is_built(ins):
    if isinstance(ins, dict):
        if 'obj' in ins and 'args' in ins:
            return False
        for i in ins.values():
            if not is_built(i):
                return False
    elif isinstance(ins, str):
        if '/' in ins:  # reference may exist in a path string.
            inss = ins.split('/')
            return is_built(inss)
        elif is_ref_type(ins):
            return False
    elif isinstance(ins, list):
        for i in ins:
            if not is_built(i):
                return False
    return True

def deep_build(ins, config, build_space: set = None):
    if is_built(ins):
        return ins

    if build_space is None:
        build_space = set()

    if isinstance(ins, list):
        for i in range(len(ins)):
            ins[i] = deep_build(ins[i], config, build_space)
        return ins
    elif isinstance(ins, dict):
        if 'obj' in ins and 'args' in ins: # return a instantiated module.
            obj = ins['obj']
            args = ins['args']
            assert isinstance(args, dict), f"Args for {obj} must be a dict."
            args = deep_build(args, config, build_space)

            module_cls = dynamic_import(obj)
            mm = module_cls(**args)
            return mm
        else:  # return a nomal dict.
            for k in ins:
                ins[k] = deep_build(ins[k], config, build_space)
            return ins
    elif isinstance(ins, str):
        if '/' in ins:  # reference may exist in a path string.
            inss = ins.split('/')
            inss = deep_build(inss, config, build_space)
            ins = '/'.join(inss)
            return ins
        elif is_ref_type(ins):
            ref = ins[1:-1]

            if ref in build_space:
                raise ValueError("Cross referencing is not allowed in config.")
            build_space.add(ref)

            assert hasattr(config, ref), f"Key name {ins} not found in config."
            attr = getattr(config, ref)
            attr = deep_build(attr, config, build_space)
            setattr(config, ref, attr)

            build_space.remove(ref)
            return attr
        else:
            return ins
    else:
        return ins

def build(name: str, config: Config):
    return deep_build(f"<{name}>", config)
