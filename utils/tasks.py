import torch
from archs.classifiers import GWFuserClassifier
from archs.segmenters import GWFuserSegUPerNet
from archs.segmenters import GWFuserRegUPerNet

def create_decoder(cfg, extraction_dims):
    base_task = cfg.task
    variant = cfg.fuser
    
    common_kwargs = {
        'feature_dims': extraction_dims,
        'projection_dim': cfg.projection_dim,
        # 'save_timesteps': cfg.save_timesteps,
        'num_classes': cfg.num_classes,
    }
    
    if base_task in ['classification', 'multi_label_classification']:
        cls_map = {
            'gw': (GWFuserClassifier, {}),
        }
    elif base_task in ['segmentation', 'change_detection']:
        common_kwargs.update({
            'pool_scales': cfg.get("pool_scales"),
            'rescales': cfg.get("rescales"),
            'channels': cfg.get("channels")
        })
        cls_map = {
            'gw': (GWFuserSegUPerNet, {}),
        }
    
    elif base_task in ['regression']:
        common_kwargs.update({
            'pool_scales': cfg.get("pool_scales"),
            'rescales': cfg.get("rescales"),
            'channels': cfg.get("channels")
        })
        cls_map = {
            'gw': (GWFuserRegUPerNet, {}),
        }
        
    else:
        raise ValueError(f"Unknown base task: {base_task}")
    
    if variant not in cls_map:
        raise ValueError(f"Unknown variant '{variant}' for task '{base_task}'")

    decoder, extra_cfg_attrs = cls_map[variant]
    kwargs = common_kwargs.copy()
    for kwarg, cfg_attr in extra_cfg_attrs.items():
        kwargs[kwarg] = cfg_attr

    return decoder(**kwargs)


def set_criterion(task_type, ignore_index=None):

    if task_type in ['segmentation', 'classification', 'change_detection']:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif task_type in ['multi_label_classification']:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task_type in ['regression']:
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Task type {task_type} not supported")
    
    return criterion