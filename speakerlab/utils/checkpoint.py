# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import yaml
import time
import pathlib
import collections
import logging
import torch

logger = logging.getLogger(__name__)

CKPT_PREFIX = "CKPT"
METAFNAME = f"{CKPT_PREFIX}.yaml"
PARAMFILE_EXT = ".ckpt"

def ckpt_recency(ckpt):
    return ckpt.meta["unixtime"]

Checkpoint = collections.namedtuple(
    "Checkpoint", ["path", "meta", "paramfiles"]
)
# Creating a hash allows making checkpoint sets
Checkpoint.__hash__ = lambda self: hash(self.path)

class Checkpointer:
    """
    This is a simplified version of checkpointer used in SpeechBrain,
    https://github.com/speechbrain/speechbrain
    """
    def __init__(
        self,
        checkpoints_dir,
        recoverables,
        allow_partial_load=False,
    ):
        self.checkpoints_dir = pathlib.Path(checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.recoverables = recoverables
        self.allow_partial_load = allow_partial_load

    def recover_if_possible(
        self,
        epoch=None,
        device=None,
    ):
        checkpoints = []
        for ckpt_dir in self._list_checkpoint_dirs():
            with open(ckpt_dir / METAFNAME) as fi:
                meta = yaml.load(fi, Loader=yaml.Loader)
            paramfiles = {}
            for ckptfile in ckpt_dir.iterdir():
                if ckptfile.suffix == PARAMFILE_EXT:
                    paramfiles[ckptfile.stem] = ckptfile
            checkpoints.append(Checkpoint(ckpt_dir, meta, paramfiles))

        if len(checkpoints) > 0:
            if epoch is None:
                logger.info("Load the recent checkpoint")
                checkpoints = sorted(checkpoints, key=ckpt_recency, reverse=True)
                chosen_ckpt = checkpoints[0]
                self.load_checkpoint(chosen_ckpt, device)
            else:
                is_found = False
                for chosen_ckpt in checkpoints:
                    if 'epoch' in chosen_ckpt.meta and chosen_ckpt.meta['epoch'] == epoch:
                        is_found = True
                        self.load_checkpoint(chosen_ckpt, device)
                        break
                if not is_found:
                    raise Exception(f"Checkpoint of epoch {epoch} not found, please check the {METAFNAME} files.")
        else:
            logger.info("Would load a checkpoint here, but none found yet.")


    def _list_checkpoint_dirs(self):
        return [
            x
            for x in self.checkpoints_dir.iterdir()
            if Checkpointer._is_checkpoint_dir(x)
        ]

    def load_checkpoint(self, checkpoint, device=None):
        logger.info(f"Loading a checkpoint from {checkpoint.path}")
        for name, obj in self.recoverables.items():
            try:
                loadpath = checkpoint.paramfiles[name]
            except KeyError:
                if self.allow_partial_load:
                    MSG = f"Loading checkpoint from {checkpoint.path}, \
                            but missing a load path for {name}"
                    warnings.warn(MSG, UserWarning)
                    continue
                else:
                    MSG = f"Loading checkpoint from {checkpoint.path}, \
                            but missing a load path for {name}"
                    raise RuntimeError(MSG)

            # Recoverable obj has to have 'load()' attr or be a torch.nn.Module
            if hasattr(obj,'load'):
                obj.load(loadpath, device)
            elif isinstance(obj, torch.nn.Module):
                state = torch.load(loadpath, map_location=device)
                obj.load_state_dict(state)
            elif hasattr(obj,'load_state_dict'):
                state = torch.load(loadpath)
                obj.load_state_dict(state)
            else:
                MSG = f"Don't know how to load {type(obj)}."
                raise RuntimeError(MSG)

    def save_checkpoint(
        self, meta={}, name=None, epoch=None
    ):
        if name is None:
            ckpt_dir = self._new_checkpoint_dirpath(epoch)
        else:
            ckpt_dir = self._custom_checkpoint_dirpath(name)
        os.makedirs(ckpt_dir)
        if epoch is not None:
            meta.update({'epoch':int(epoch)})
        self._save_checkpoint_metafile(
            ckpt_dir / METAFNAME, meta,
        )
        saved_paramfiles = {}
        for name, obj in self.recoverables.items():
            objfname = f"{name}" + PARAMFILE_EXT
            savepath = ckpt_dir / objfname
            saved_paramfiles[name] = savepath

            # Saved obj has to have 'save()' attr or be a torch.nn.Module
            if hasattr(obj, 'save'):
                obj.save(savepath)
            elif isinstance(obj, torch.nn.Module):
                state_dict = obj.state_dict()
                torch.save(state_dict, savepath)
            elif hasattr(obj, 'state_dict'):
                state_dict = obj.state_dict()
                torch.save(state_dict, savepath)
            else:
                MSG = f"Don't know how to save {type(obj)}."
                raise RuntimeError(MSG)
        logger.info(f"Saved a checkpoint in {ckpt_dir}")

    def _new_checkpoint_dirpath(self, epoch=None):
        if epoch is not None:
            stamp = f"EPOCH-{epoch}"
        else:
            t = time.time()
            stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(t))
        suffix_num = 0
        while (
            self.checkpoints_dir / f"{CKPT_PREFIX}-{stamp}-{suffix_num:02d}"
        ).exists():
            suffix_num += 1
        return self.checkpoints_dir / f"{CKPT_PREFIX}-{stamp}-{suffix_num:02d}"

    def _custom_checkpoint_dirpath(self, name):
        return self.checkpoints_dir / f"{CKPT_PREFIX}+{name}"

    def _save_checkpoint_metafile(
        self, fpath, meta_to_include={},
    ):
        meta = {"unixtime": time.time()}
        meta.update(meta_to_include)
        with open(fpath, "w") as fo:
            fo.write(yaml.dump(meta))
        return meta

    @staticmethod
    def _is_checkpoint_dir(path):
        path = pathlib.Path(path)
        if not path.is_dir():
            return False
        if not path.name.startswith(CKPT_PREFIX):
            return False
        return (path / METAFNAME).exists()
