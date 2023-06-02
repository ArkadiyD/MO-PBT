import math
import random
import numpy as np
import pickle
import torch
import time
import os
from torchvision import transforms as transforms

from models import make_function_create_model

def createAndCleanFolder(folder):
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(folder)
    f = 'collectedData.txt'
    if f in files:
        file = open(folder + '/' + f, 'r')
        file.readlines()

    for f in files:
        f = folder + '/' + f

        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

    return True


def delete_old_model_files(base_folder, model_id, epoch):
    model_files = os.listdir('%s/models' % base_folder)
    for file_name in model_files:
        model_id_ = int(file_name.split('_')[1])
        if model_id_ != model_id:
            continue
        file_epoch = int(file_name.split('_')[2])
        if file_epoch < epoch:
            print(f'delete {file_name}')
            os.remove('%s/models/%s' % (base_folder, file_name))

def dict_to_cuda(d):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

def adjust_optimizer_settings(optimizer, lr, momentum=None, wd=None, nesterov=None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if momentum is not None:
            param_group['momentum'] = momentum
        if wd is not None:
            param_group['weight_decay'] = wd
        if nesterov is not None:
            param_group['nesterov'] = nesterov

    return optimizer


class MySubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
            indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices
        print(self.indices)

    def __iter__(self):
        # print(torch.randperm(len(self.indices)))
        return (self.indices[i] for i in np.random.permutation(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()+worker_id
    torch_seed = torch_seed % 2**30

    random.seed(torch_seed)
    np.random.seed(torch_seed)


def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_rng_state():
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state()
    np_state = np.random.get_state()
    python_state = random.getstate()
    return torch_state, torch_cuda_state, np_state, python_state


def set_rng_state(rng_state):
    torch_state, torch_cuda_state, np_state, python_state = rng_state
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state(torch_cuda_state)
    np.random.set_state(np_state)
    random.setstate(python_state)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    #print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def my_argsort(seq):
    if isinstance(seq[0], tuple) or isinstance(seq[0], list) or isinstance(seq[0], np.ndarray):
        return [x for x, y in sorted(enumerate(list(seq)), key=lambda x: (x[1][0], x[1][1]))]

    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]

def get_function_create_model(args, data_provider):
    kwargs_for_function_create_model = vars(args)
    if 'n_channels' not in kwargs_for_function_create_model['model_parameters']:
        kwargs_for_function_create_model['model_parameters']['n_channels'] = 3
    kwargs_for_function_create_model['model_parameters']['data_provider'] = data_provider
    try:
        kwargs_for_function_create_model['model_parameters']['d_num'] = data_provider.d_num
        kwargs_for_function_create_model['model_parameters']['cat'] = data_provider.cat
    except Exception:
        pass
    function_create_model = make_function_create_model(
        **kwargs_for_function_create_model)

    return function_create_model