from functools import partial

from models.wide_resnet import WideResNet
from models.FTTransformer import *

def make_function_create_model(**kwargs):
    if kwargs['model_class'] == 'WideResNet':
        return partial(make_wide_resnet, **kwargs['model_parameters'])

    elif kwargs['model_class'] == 'FTTransformer':
        return partial(make_FTTransformer, **kwargs['model_parameters'])

    raise ValueError('model_class type should be one of [\'WideResNet\', \'FTTransformer\'], found:%s' % kwargs['model_class'])


def make_wide_resnet(depth, widen_factor, dropout_rate, data_provider, **kwargs):
    return WideResNet(depth, widen_factor, dropout_rate, data_provider.n_classes)

def make_FTTransformer(dropout, d_num, cat, **kwargs):
    return FTTransformer(d_num, cat, dropout, **kwargs)


