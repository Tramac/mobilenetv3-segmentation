import os
from .segmentation import *


def get_segmentation_model(model, **kwargs):
    models = {
        'mobilenetv3_large': get_mobilenetv3_large_seg,
        'mobilenetv3_small': get_mobilenetv3_small_seg,
    }
    return models[model](**kwargs)


def get_model_file(name, root='~/.torch/models'):
    root = os.path.expanduser(root)
    file_path = os.path.join(root, name + '.pth')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')
