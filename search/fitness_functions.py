import os.path

import torch

from search.hyperparameters import *
from search.train_and_evaluate import *
from utils import *


class HyperparametersSearch:
    def __init__(self, folder, max_epochs, epoch_step, data_provider, hyperparameters, function_create_model):
        self.folder = folder
        self.max_epochs = max_epochs
        self.epoch_step = epoch_step
        self.hyperparameters = hyperparameters
        self.data_provider = data_provider
        self.function_create_model = function_create_model
        self.device = 'cuda'

    def try_load_checkpoint(self, model_id, epoch):
        model_files = os.listdir('%s/models' % self.folder)
        model_files = [model_file for model_file in model_files if int(
            model_file.split('_')[1]) == model_id]

        model_files = sorted(model_files, key=lambda x: (int(x.split('_')[2])))
        model_files = [model_file for model_file in model_files if int(
            model_file.split('_')[2]) == epoch]

        if len(model_files) != 1:
            return None

        model_file = model_files[0]
        checkpoint = torch.load('%s/models/%s' %
                                (self.folder, model_file), map_location='cpu')
        print('loaded model from epoch %d' % epoch)
        return checkpoint

    def fitness(self, encoded_solution, model_id, epoch, weights_seed=None, train_seed=None, save_model=True, cur_data_provider=None):
        if self.data_provider is not None:
            data_provider = self.data_provider
        elif cur_data_provider is not None:
            data_provider = cur_data_provider
        else:
            raise Exception("data provider is not None!")

        torch.cuda.empty_cache()

        encoded_solution = tuple([int(x) for x in encoded_solution])

        self.hyperparameters.convert_encoding_to_hyperparameters(encoded_solution)

        if weights_seed is not None:
            set_random_seeds(weights_seed)
        net = self.function_create_model(
            data_provider=data_provider, hyperparameters=self.hyperparameters)
        self.optimizer = self.hyperparameters.get_optimizer(net)
        if isinstance(net, FTTransformer):
            net.update(self.hyperparameters.architecture['dropout'])

        if train_seed is None:
            set_random_seeds(int(time.time()))
        else:
            set_random_seeds(train_seed)
        
        checkpoint = self.try_load_checkpoint(model_id, epoch)
        if checkpoint is not None:
            net = checkpoint['model']
            if isinstance(net, FTTransformer):
                net.update(self.hyperparameters.architecture['dropout'])

            self.optimizer = checkpoint['optimizer']

            if self.hyperparameters.if_search_wd:
                lr, wd, momentum, nesterov = checkpoint['hyperparameters'].get_optimizer_params(
                )
                print(
                    f'optimizer loaded from checkpoint: {lr=} ; {wd=} ; {momentum=} ; {nesterov=}')
                lr, wd, momentum, nesterov = self.hyperparameters.get_optimizer_params()
                self.optimizer = adjust_optimizer_settings(self.optimizer, lr=lr, wd=wd, nesterov=nesterov,
                                                           momentum=momentum)
            else:
                lr, wd, momentum, nesterov = checkpoint['hyperparameters'].get_optimizer_params(
                )
                print(
                    f'optimizer loaded from checkpoint: {lr=} ; {wd=} ; {momentum=} ; {nesterov=}')
                self.hyperparameters.optimizer['lr'] = lr

            optimizer_to(self.optimizer, self.device)

        net = net.cuda()

        info_train_and_val = train(net, self.hyperparameters, self.optimizer, data_provider, epoch,
                                    epoch + self.epoch_step, self.max_epochs, self.device)
        try:
            net_state = net.state_dict()
            optimizer_state = self.optimizer.state_dict()

            val_score = info_train_and_val['val_score']
            test_score = info_train_and_val['test_score']

        except Exception as e:
            print(e)

        print('saving model... :')
        print(val_score, test_score, '%s/models/model_%d_%d' %
                (self.folder, model_id, epoch + self.epoch_step))

        if checkpoint is not None:
            prev_schedule = checkpoint['config']
            prev_scores = checkpoint['cur_scores']
            model_ids = checkpoint['model_ids']
        else:
            prev_schedule = []
            prev_scores = []
            model_ids = []

        prev_schedule.append(tuple(encoded_solution))
        prev_scores.append(tuple([val_score, test_score]))
        model_ids.append(model_id)

        torch.save({
            'cur_scores': prev_scores,
            'model_ids': model_ids,
            'model_state_dict': net_state,
            'model': net,
            'hyperparameters': self.hyperparameters,
            'config': prev_schedule,
            'optimizer_state_dict': optimizer_state,
            'optimizer': self.optimizer,

        }, '%s/models/model_%d_%d' % (self.folder, model_id, epoch + self.epoch_step))

        return val_score[-1], test_score[-1], info_train_and_val
    
        