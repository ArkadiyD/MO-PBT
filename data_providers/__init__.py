from data_providers.cifar import *
from data_providers.higgs import *
from data_providers.adult import *
from data_providers.adult_fairness import *
from data_providers.celeba_fairness import *
from data_providers.clickprediction import ClickPredictionDataProvider


def get_data_provider(**kwargs):
    if kwargs['dataset_name'] == 'cifar10':
        data_provider = CIFAR10DataProvider(**kwargs['dataset_parameters'])
    elif kwargs['dataset_name'] == 'cifar100':
        data_provider = CIFAR100DataProvider(**kwargs['dataset_parameters'])

    elif kwargs['dataset_name'] == 'CelebaFairness':
        data_provider = CelebaFairnessDataProvider(
            **kwargs['dataset_parameters'])

    elif kwargs['dataset_name'] == 'AdultFairness':
        data_provider = AdultFairnessDataProvider(
            **kwargs['dataset_parameters'])

    elif kwargs['dataset_name'] == 'Higgs':
        data_provider = HiggsDataProvider(**kwargs['dataset_parameters'])
    elif kwargs['dataset_name'] == 'Adult':
        data_provider = AdultDataProvider(**kwargs['dataset_parameters'])
    elif kwargs['dataset_name'] == 'ClickPrediction':
        data_provider = ClickPredictionDataProvider(
            **kwargs['dataset_parameters'])

    else:
        raise ValueError('dataset_name type should be one of [\'cifar10\', \'cifar100\', \'CelebaFairness\', \'AdultFairness\', \'Higgs\', \'Adult\', \'ClickPrediction\'], found:%s' % kwargs['dataset_name'])

    return data_provider
