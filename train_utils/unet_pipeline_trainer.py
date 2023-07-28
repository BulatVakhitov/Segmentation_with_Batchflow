import os
import numpy as np

from batchflow.batchflow import B, M, V, Notifier, Pipeline
from batchflow.batchflow.models.torch import TorchModel

import torch

from PIL import Image

from .utils import mIoU


def train(
    dataset,
    config,
    loss,
    optimizer,
    train_params,
    n_iters=1000,
    visible_devices='0,1',
    notifier_config=None,
    frequency=100,
    prefetch=4
):
    """
    Creates, trains and returns pipeline based on passed parameters

    Parameters
    ----------

    dataset: Batchflow.Dataset
        dataset which is splited for train, test and validation part
    config: dict
        models config
    loss: dict or nn.Module or function
        segmentation loss. Can be either dict in bf.config style, nn.Module or function
    optimizer: dict
        dict with at least two fields: name(str, nn.Module) and lr
    train_params: dict
        dict with the following fields: 'BATCH_SIZE', 'LR', 'NUM_CLASSES', 'NUM_EPOCHS',
        'IMAGE_SHAPE'
    visible_devices: str
        gpu ids that will be visible in this experiment. Should be string of gpu ids
        separated by comma
    notifier_config: dict
        plotting configuration

    Returns
    -------

    train_val_pipeline: Pipeline
        trained pipeline
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    config['classes'] = train_params['NUM_CLASSES']
    config['inputs_shapes'] = (3, *train_params['IMAGE_SHAPE'])
    config['target_shapes'] = (1, *train_params['IMAGE_SHAPE'])
    config['head'] = {
        'channels': train_params['NUM_CLASSES'],
        'layout': 'c',
        'output_type': 'tensor'
    }
    config['loss'] = loss
    config['optimizer'] = optimizer
    config['device'] = [torch.device('cuda:'+str(device)) for device in range(torch.cuda.device_count())]

    preprocessing_pipeline = (
        Pipeline()
        .resize(
            size=train_params['IMAGE_SHAPE'],
            resample=Image.Resampling.BILINEAR,
            src='images',
            dst='images'
        )
        .to_array(channels='first', dtype=np.float32, src='images', dst='images')
        .resize(
            size=train_params['IMAGE_SHAPE'],
            resample=Image.Resampling.NEAREST,
            src='labels',
            dst='masks'
        )
        .to_array(channels='first', dtype=np.int64, src='masks', dst='masks')
    )

    train_model_pipeline = (
        Pipeline()
        .init_model('model', model_class=TorchModel, config=config)
        .train_model('model', inputs=B('images'), targets=B('masks'))
    )

    train_pipeline = (
        preprocessing_pipeline + \
        train_model_pipeline
    ) << dataset.train

    val_model_pipeline = (
        Pipeline()
        .import_model('model', train_pipeline)
        .init_variable('miou_micro', [])
        .init_variable('miou_macro', [])
        .init_variable('val_loss', [])
        .predict_model(
            'model',
            inputs=B('images'),
            outputs=['predictions', 'loss'],
            targets=B('masks'),
            save_to=[B('predictions'), V('val_loss', mode='a')],
            transfer_from_device=False
        )
        .mIoU(
            preds=B('predictions'),
            mask=B('masks'),
            num_classes=train_params['NUM_CLASSES'],
            multiclass='micro',
            save_to=V('miou_micro', mode='a')
        )
        .mIoU(
            preds=B('predictions'),
            mask=B('masks'),
            num_classes=train_params['NUM_CLASSES'],
            multiclass='macro',
            save_to=V('miou_macro', mode='a')
        )
    )

    val_pipeline = (
        preprocessing_pipeline + \
        val_model_pipeline
    ) << dataset.validation

    if notifier_config is None:
        notifier_config= {
            'bar': 'n',
            'frequency': frequency,
            'graphs': [
                M.model.loss_list,
                'val_loss',
                'miou_micro',
                'miou_macro',
                'gpu_memory',
                #'gpu_memory_utilization'
            ],
            'plot_config': {
                'ncols': 3,
                'nrows': 2,
                'figsize': (25, 15),
                'title_size': 15,
                'legend_size': 10,
                'alpha': 0.6
            },
            'n_iters': n_iters
        }

    notifier = Notifier(**notifier_config)

    for i in range(n_iters):
        train_batch = train_pipeline.next_batch(
            batch_size=train_params['BATCH_SIZE'],
            prefetch=prefetch
        )
        val_batch = val_pipeline.next_batch(
            batch_size=train_params['BATCH_SIZE'],
            notifier=notifier,
            prefetch=prefetch//4
        )
    return train_pipeline, val_pipeline
