from torch import optim
import deepdiff
from augmentations.augmentation_policy import *
from augmentations.pil_augmentations import *
import torchvision.transforms as T

from search.train_and_evaluate import *
from search.loss import *
import utils

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class Hyperparameters:
	def __init__(self, num_classes, **kwargs):
		self.args = kwargs

		self.num_classes = num_classes
		self.train_batch_size = kwargs['dataset_parameters']['train_batch_size']
		self.test_batch_size = kwargs['dataset_parameters']['test_batch_size']

		if 'resolution' in kwargs['dataset_parameters']:
			self.resolution = kwargs['dataset_parameters']['resolution']

		self.dataset_name = kwargs['dataset_name'].lower()

		self.task_type = kwargs['task_type']
		self.loss_name = kwargs['loss']['name']

		self.if_search_augs = kwargs['search_space_parameters']['if_search_augs']
		self.if_search_wd = kwargs['search_space_parameters']['if_search_wd']

		self.if_search_dropout = kwargs['search_space_parameters']['if_search_dropout']

		self.if_search_loss = kwargs['search_space_parameters']['if_search_loss']

		self.scheduler = kwargs['optimizer']['scheduler']

		self.optimizer = {'optimizer_name': kwargs['optimizer']['optimizer_name'], 'lr': float(
			kwargs['optimizer']['lr_value']), 'weight_decay': float(kwargs['optimizer']['wd_value']), 'momentum': float(kwargs['optimizer']['momentum'])}

		self.loss = {'name': kwargs['loss']['name']}

		self.augmentations = {}

		self.augmentations['policy'] = kwargs['dataset_parameters']['policy_type']

		if self.augmentations['policy'] == 'RandAugment':
			self.augmentations_search_space = RandAugmentSearchSpace(
				mag_bins=10)

		elif self.augmentations['policy'] == 'PBA':
			self.augmentations_search_space = PBASearchSpace(mag_bins=10)

		elif self.augmentations['policy'] == 'noaug':
			self.augmentations_search_space = None

		else:
			raise ValueError('dataset_parameters[\'policy_type\'] should be one of [\'RandAugment\', \'PBA\', \'noaug\'], found:%s ' % self.augmentations['policy'])

		if 'dropout' in kwargs['model_parameters']:
			self.architecture = {'dropout': float(
				kwargs['model_parameters']['dropout'])}
		else:
			self.architecture = {}

		if 'resolution' in kwargs['dataset_parameters']:
			self.resolution = self.min_resolution = int(
				kwargs['dataset_parameters']['resolution'])
		else:
			self.resolution = None

		self.loss['loss_weight'] = 1.0
		
		self.alphabet, self.sizes = self.get_search_space_sizes()

	def get_search_space(self):
		search_space = {}
		if self.if_search_wd:
			search_space['weight_decay'] = {'weight_decay': [0.0]+list(np.power(10, np.linspace(-6, -1, 9)))}
		
		if self.if_search_dropout:
			search_space['dropout'] = {'dropout_%d' % k: tuple(np.linspace(0.0, 0.8, 10)) for k in range(3)}
		
		if self.if_search_loss:
			if self.loss_name == 'CESPLoss':
				search_space['loss_weight'] = {'loss_weight': [
					0.0]+list(np.power(10, np.linspace(-2, 1, 9)))}
			elif self.loss_name == 'WeightedCE':
				search_space['loss_weight'] = {
					'loss_weight': np.linspace(0.1, 0.9, 10)}
			elif self.loss_name == 'AdvLoss':
				search_space['loss_weight'] = {'loss_weight': [
					0.0]+list(np.power(10, np.linspace(-1, 1, 9)))}
			else:
				raise ValueError('loss[\'name\'] should be one of [\'WeightedCE\', \'CESPLoss\', \'ApplyTradesLoss\'], found:%s '% self.loss_name)

		if self.if_search_augs:
			if self.augmentations_search_space is None:
				pass
			else:
				search_space['augmentations'] = self.augmentations_search_space.search_space

		print(f'{search_space=}')
		return search_space

	def get_search_space_sizes(self):
		self.search_space = self.get_search_space()
		self.sizes = []
		for x in self.search_space:
			for x in range(len(self.search_space[x])):
				self.sizes.append(0)

		self.alphabet = []
		cnt = 0
		if self.if_search_augs:

			if self.augmentations['policy'] == 'RandAugment' or self.augmentations['policy'] == 'PBA':
				aug_options = self.search_space['augmentations']
				for aug in aug_options:
					for s in aug_options[aug]:
						self.sizes[cnt] += 1
						self.alphabet.append(len(s))
					cnt += 1

			elif self.augmentations['policy'] == 'noaug':
				pass

		if self.if_search_wd:
			options = self.search_space['weight_decay']
			for opt in options:
				self.sizes[cnt] += 1
				self.alphabet.append(len(options[opt]))
				cnt += 1

		if self.if_search_dropout:
			options = self.search_space['dropout']
			for opt in options:
				self.sizes[cnt] += 1
				self.alphabet.append(len(options[opt]))
				cnt += 1

		if self.if_search_loss:
			options = self.search_space['loss_weight']
			for opt in options:
				self.sizes[cnt] += 1
				self.alphabet.append(len(options[opt]))
				cnt += 1

		return self.alphabet, self.sizes

	def get_transforms(self):
		if self.augmentations['policy'] == 'noaug':
			if self.dataset_name != 'celebafairness' and not ('cifar' in self.dataset_name):
				return None

		if self.dataset_name == 'celebafairness':
			mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
		else:
			mean, std = _CIFAR_MEAN, _CIFAR_STD
		policy = ''

		transforms_dict = {'train': None, 'test': None}
		for type in transforms_dict:
			if type == 'train':

				transform = self.create_custom_transform(
					self.augmentations['policy'], mean, std)
				transforms_dict[type] = transform

			else:
				transform = self.create_custom_transform_val(
					self.augmentations['policy'], mean, std)
				transforms_dict[type] = transform

		return transforms_dict

	def get_optimizer(self, net):
		if self.optimizer['optimizer_name'] == 'SGDNesterov':
			return getattr(optim, 'SGD')(net.parameters(), lr=abs(self.optimizer['lr']),
										 weight_decay=self.optimizer['weight_decay'],
										 momentum=self.optimizer['momentum'], nesterov=True)

		elif self.optimizer['optimizer_name'] == 'SGD':
			return getattr(optim, 'SGD')(net.parameters(), lr=self.optimizer['lr'],
										 weight_decay=self.optimizer['weight_decay'],
										 momentum=self.optimizer['momentum'], nesterov=False)

		elif self.optimizer['optimizer_name'] == 'AdamW':
			return getattr(optim, 'AdamW')(net.parameters(), lr=self.optimizer['lr'],
										   weight_decay=self.optimizer['weight_decay'])

		else:
			raise ValueError('optimizer_name should be one of [\'AdamW\', \'SGD\', \'SGDNesterov\'], found: %s' % self.optimizer['optimizer_name'])


	def get_optimizer_name(self):
		return self.optimizer['optimizer_name']

	def get_optimizer_params(self):
		nesterov = None
		momentum = None
		if self.optimizer['optimizer_name'] == 'SGD':
			nesterov = False
			momentum = self.optimizer['momentum']
		elif self.optimizer['optimizer_name'] == 'SGDNesterov':
			nesterov = True
			momentum = self.optimizer['momentum']

		return self.optimizer['lr'], self.optimizer['weight_decay'], momentum, nesterov

	def create_custom_transform(self, policy_type, norm_mean=None, norm_var=None):
		if policy_type == 'noaug':
			if self.dataset_name == 'celebafairness':
				augs_list = [T.ToTensor(), T.Normalize(
					[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
				return MyPILTransformsCompose(augs_list)
			elif 'cifar' in self.dataset_name:
				augs_list = [transforms.RandomCrop(self.resolution, padding=4), transforms.RandomHorizontalFlip(
					p=0.5), T.ToTensor(), T.Normalize(norm_mean, norm_var)]
				if self.loss_name == 'AdvLoss':
					augs_list.pop()
				return MyPILTransformsCompose(augs_list)
			else:
				return None

		augs_list = []

		if policy_type == 'RandAugment' or policy_type == 'PBA':
			if policy_type == 'RandAugment':
				policy = AugmentationPolicyRandAugment(
					self.augmentations['augmentations_list'], self.augmentations['n'], self.augmentations['m'])
			else:
				policy = AugmentationPolicyPBA(self.augmentations['augmentations_list'], self.augmentations['n_ops_probs'])
            
			augs_list = [policy]
			if 'cifar' in self.dataset_name:
				augs_list.append(transforms.RandomCrop(
					self.resolution, padding=4))
				augs_list.append(transforms.RandomHorizontalFlip(p=0.5))
			elif self.dataset_name == 'celebafairness':
				augs_list = [transforms.Resize(self.resolution)] + augs_list

		
		augs_list += [T.ToTensor(), T.Normalize(norm_mean, norm_var)]
		if self.loss_name == 'AdvLoss':
			augs_list.pop()

		if 'cutout' in self.augmentations and self.augmentations['cutout']['prob'] > 0:
			augs_list.append(TensorCutout(
				self.augmentations['cutout']['prob'], self.augmentations['cutout']['mag']))

		print('train augs_list', augs_list)
		return MyPILTransformsCompose(augs_list)

	def create_custom_transform_val(self, policy_type, norm_mean, norm_var):

		augs_list = []

		if self.dataset_name == 'celebafairness':
			augs_list = [transforms.Resize(self.resolution)]

		augs_list += [T.ToTensor(), T.Normalize(norm_mean, norm_var)]

		if self.loss_name == 'AdvLoss':
			augs_list.pop()
		print('val augs_list', augs_list)
		return MyPILTransformsCompose(augs_list)

	def convert_encoding_to_hyperparameters(self, encoding):
		total_size = np.sum(self.sizes)
		assert len(encoding) == total_size
		encoding = [int(x) for x in encoding]
		cnt = 0
		if self.augmentations['policy'] == 'RandAugment':
			self.augmentations['n'] = self.augmentations_search_space.search_space['n'][0][encoding[cnt]]
			self.augmentations['m'] = self.augmentations_search_space.search_space['m'][0][encoding[cnt+1]]
			cnt += 2

			self.augmentations['cutout'] = {"prob": self.augmentations_search_space.search_space['special_Cutout'][0][encoding[cnt]],
											"mag": self.augmentations_search_space.search_space['special_Cutout'][1][encoding[cnt+1]]
											}
			cnt += 2

			self.augmentations['augmentations_list'] = self.augmentations_search_space.augs_list

		elif self.augmentations['policy'] == 'PBA':
			augs_dict = self.augmentations_search_space.search_space
			augs_list_ = sorted(list(augs_dict.keys()))
			augs_list = sorted(
				[x for x in augs_list_ if 'op_' not in x and 'special' not in x])
			augs_list_ops = sorted([x for x in augs_list_ if 'op_' in x])

			self.augmentations['augmentations_list'] = []
			for op_name in augs_list:
				prob = self.augmentations_search_space.search_space[op_name][0][encoding[cnt]]
				mag = 0

				if len(self.augmentations_search_space.search_space[op_name]) == 2:
					mag = self.augmentations_search_space.search_space[op_name][1][encoding[cnt+1]]

				self.augmentations['augmentations_list'].append(
					(op_name, prob, mag))
				cnt += len(
					self.augmentations_search_space.search_space[op_name])
			
			self.augmentations['cutout'] = {"prob": self.augmentations_search_space.search_space['special_Cutout'][0][encoding[cnt]],
											"mag": self.augmentations_search_space.search_space['special_Cutout'][1][encoding[cnt+1]]
											}
			cnt += 2

			self.augmentations['n_ops_probs'] = []
			for op_name in augs_list_ops:
				prob = self.augmentations_search_space.search_space[op_name][0][encoding[cnt]]
				self.augmentations['n_ops_probs'].append(prob)
				cnt += 1

			print(f'{self.augmentations=}')

		if self.if_search_wd:
			opt_dict = self.search_space['weight_decay']
			current_option = encoding[cnt]
			self.optimizer['weight_decay'] = opt_dict['weight_decay'][current_option]
			cnt += 1

		if self.if_search_dropout:
			self.architecture['dropout'] = []
			for opt in sorted(self.search_space['dropout']):
				self.architecture['dropout'].append(
					self.search_space['dropout'][opt][encoding[cnt]])
				cnt += 1

		if self.if_search_loss:
			opt_dict = self.search_space['loss_weight']
			current_option = encoding[cnt]
			self.loss['loss_weight'] = opt_dict['loss_weight'][current_option]
			cnt += 1

		assert cnt == len(encoding) == total_size


