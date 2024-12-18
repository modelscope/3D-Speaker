import os
import argparse
import pathlib
import re
import sys

import torch

sys.path.append('%s/../..'%os.path.dirname(__file__))

from speakerlab.utils.builder import build
from speakerlab.utils.utils import get_logger
from speakerlab.utils.config import build_config
from speakerlab.utils.fileio import load_wav_scp
from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path


logger = get_logger()


def get_args():
    parser = argparse.ArgumentParser(
        description="Export exist checkpoint to onnx file"
    )
    parser.add_argument(
        "--experiment_path", required=True, type=str,
        help="Your experiment path, we could download or save something in this path, "
        "or you have trained your model using 3D-Speaker"
    )
    parser.add_argument(
        "--model_id", default=None,
        help="The model id from modelscope. "
        "If passed, try to download the model from modelscope and save to experiments path. "
        "Else, try to find checkpoint under `experiment_path`"
    )
    parser.add_argument(
        "--target_onnx_file", required=True, help="The target onnx file"
    )
    return parser.parse_args()


# TODO: Load file to get these informations
# TODO: Support more models
# Please note you can export your own model which could not in this dict.
onnx_supports_dict = {
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'model_pt': 'campplus_cn_common.bin',
    },
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5',
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    # ERes2Net_Base trained on 200k labeled speakers
    'iic/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
            'args': {
                'feat_dim': 80,
                'embedding_size': 512,
                'm_channels': 32,
            },
        },
        'model_pt': 'pretrained_eres2net.pt',
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0',
        'model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'model_pt': 'campplus_cn_en_common.pt',
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 512,
            },
        },
        'model_pt': 'campplus_voxceleb.bin',
    },
    # ERes2Net trained on VoxCeleb
    'iic/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    # ERes2Net_Base trained on 3dspeaker
    'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1',
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
            'args': {
                'feat_dim': 80,
                'embedding_size': 512,
                'm_channels': 32,
            },
        },
        'model_pt': 'eres2net_base_model.ckpt',
    },
    # ERes2Net_large trained on 3dspeaker
    'iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
            'args': {
                'feat_dim': 80,
                'embedding_size': 512,
                'm_channels': 64,
            },
        },
        'model_pt': 'eres2net_large_model.ckpt',
    },
    # ERes2NetV2 trained on 200k labeled speakers
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },
    # ERes2NetV2 trained on 200k labeled speakers (w24s4ep4)
    'iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1',
        'model': {
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
                "baseWidth": 24,
                "scale":4,
                "expansion": 4                
            },
        },
        'model_pt': 'pretrained_eres2netv2w24s4ep4.ckpt',
    },
}


def export_onnx_file(model, target_onnx_file):
    # build dummy input for export
    # Note: 1. feature_dim is fixed and you may change it for your own model.
    #       2. The model input shape is (batch_size, frame_num, feature_dim).
    dummy_input = torch.randn(1, 345, 80)
    torch.onnx.export(model,
                      dummy_input,
                      target_onnx_file,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['feature'],
                      output_names=['embedding'],
                      dynamic_axes={'feature': {0: 'batch_size', 1: 'frame_num'},
                                    'embedding': {0: 'batch_size'}})
    logger.info(f"Export model onnx to {target_onnx_file} finished")


def build_model_from_modelscope_id(model_id: str, local_model_path):
    logger.info(f"Build model from modelscope model_id: {model_id}")
    if not is_official_hub_path(model_id):
        raise ValueError(f"Invalid modelscope model id {model_id}")

    if model_id not in onnx_supports_dict:
        raise ValueError(
            f"model_id {model_id} is not currently onnx-supported")

    model_name = model_id.split("/")[1]

    save_dir = os.path.join(local_model_path, "pretrained", model_name)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model_info_item = onnx_supports_dict[model_id]
    cache_dir = snapshot_download(
        model_id, revision=model_info_item['revision'])
    cache_dir = pathlib.Path(cache_dir)

    # add symlink from cache_dir to save_dir.
    download_files = ['examples', model_info_item['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    # load model and checkpoint
    pretrained_model = os.path.join(save_dir, model_info_item['model_pt'])
    pretrained_state = torch.load(pretrained_model, map_location='cpu')
    model = model_info_item['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    return embedding_model


def build_model_from_custom_work_path(local_model_path):
    logger.info(f"Build model from local model path {local_model_path}")
    config_file = os.path.join(local_model_path, 'config.yaml')
    config = build_config(config_file)

    embedding_model = build('embedding_model', config)
    config.checkpointer['args']['checkpoints_dir'] = os.path.join(
        local_model_path, 'models'
    )
    config.checkpointer['args']['recoverables'] = {
        'embedding_model': embedding_model
    }
    checkpointer = build('checkpointer', config)
    checkpointer.recover_if_possible(epoch=config.num_epoch, device='cpu')

    embedding_model.eval()

    return embedding_model


def main():
    args = get_args()
    logger.info(f"{args}")

    model_id = args.model_id
    experiment_path = args.experiment_path
    target_onnx_file = args.target_onnx_file
    if model_id is not None:
        speaker_embedding_model = build_model_from_modelscope_id(
            model_id, experiment_path
        )
    else:
        speaker_embedding_model = build_model_from_custom_work_path(
            experiment_path
        )
    
    logger.info(f"Load speaker embedding finished, export to onnx")
    export_onnx_file(speaker_embedding_model, target_onnx_file)


if __name__ == '__main__':
    main()

