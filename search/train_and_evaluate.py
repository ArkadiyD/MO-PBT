from models.FTTransformer import *
from search.metric import *
from search.loss import *
import time

import tabulate
from data_providers.cifar import CIFARBaseDataProvider
from utils import *
from search.hyperparameters import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


def train(model, hyperparameters, optimizer, data_provider: CIFARBaseDataProvider, first_epoch, last_epoch,
		  max_epochs, device):
	print(f'{first_epoch=} ; {last_epoch=} ; {max_epochs=}')

	transforms = hyperparameters.get_transforms()

	if hyperparameters.loss['name'] == 'WeightedCE':
		criterion = criterion_val = torch.nn.CrossEntropyLoss(weight=torch.Tensor(
			[hyperparameters.loss['loss_weight'], 1.0-hyperparameters.loss['loss_weight']]).float().cuda())
	elif hyperparameters.loss['name'] == 'CESPLoss':
		criterion = criterion_val = CESPLoss(
			mu=hyperparameters.loss['loss_weight'])
	elif hyperparameters.loss['name'] == 'AdvLoss':
		criterion = criterion_val = ApplyTradesLoss(
			beta=hyperparameters.loss['loss_weight'], perturb_steps=hyperparameters.args['loss']['steps'], use_autocast=hyperparameters.args['use_autocast'])
	else:
		raise ValueError('loss[\'name\'] should be one of [\'WeightedCE\', \'CESPLoss\', \'ApplyTradesLoss\'], found:%s '% hyperparameters.loss['loss_weight'])

	data_loaders = data_provider.create_dataloaders(
		hyperparameters, transforms)

	columns = ['epoch time', 'overall training time', 'epoch', 'lr',
			   'val_score', 'test_score']

	all_values = {}
	all_values['epoch'] = []
	all_values['lr'] = []

	all_values['val_score'] = []
	all_values['test_score'] = []

	print('Start training...')

	scaler = torch.cuda.amp.GradScaler()
	time_start = time.time()

	for epoch in range(first_epoch, last_epoch):
		time_ep = time.time()

		print(f'{epoch=}')
		try:
			train_res = train_epoch(device, data_loaders['train'], model, criterion, optimizer,
									scaler, hyperparameters, epoch, max_epochs, last_epoch-first_epoch, hyperparameters.args)
		except Exception as e:
			print(e)

		values = [epoch + 1, train_res['lr'],
				  train_res['loss'], train_res['score']]

		if epoch == last_epoch-1:

			all_values['epoch'].append(epoch+1)
			all_values['lr'].append(train_res['lr'])

			val_res = evaluate(hyperparameters.task_type, device,
							   data_loaders['val'], model, hyperparameters, criterion_val, hyperparameters.args)
			if epoch == last_epoch-1:
				test_res = evaluate(hyperparameters.task_type, device,
									data_loaders['test'], model, hyperparameters, criterion_val, hyperparameters.args)

			try:
				all_values['val_score'].append(val_res['score'])
				if epoch == last_epoch-1:
					all_values['test_score'].append(test_res['score'])
					values += [val_res['score'], test_res['score']]
			except Exception as e:
				print(e)
			
		overall_training_time = time.time()-time_start
		values = [time.time()-time_ep, overall_training_time] + values
		table = tabulate.tabulate(
			[values], columns, tablefmt='simple', floatfmt='8.4f')
		print(table)

	del criterion

	return all_values


def train_epoch(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args):
	if hyperparameters.task_type == 'fairness':
		return train_epoch_fairness(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args)
	elif hyperparameters.task_type == 'adv_classification':
		return train_epoch_classification(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args)
	elif hyperparameters.task_type == 'tabular_classification':
		return train_epoch_tabular(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args)
	else:
		raise ValueError('task_type should be one of [\'tabular_classification\', \'adv_classification\', \'fairness\'], found:%s' % hyperparameters.task_type)

def evaluate(task, device, loader, model, hyperparameters, criterion, args):
	if task == 'fairness':
		return evaluate_fairness(device, loader, model, hyperparameters, criterion, args)
	elif task == 'adv_classification':
		return evaluate_adv_classification(device, loader, model, hyperparameters, criterion, args)
	elif task == 'tabular_classification':
		return evaluate_tabular_classification(device, loader, model, hyperparameters, criterion, args)
	else:
		raise ValueError('task_type should be one of [\'tabular_classification\', \'adv_classification\', \'fairness\'], found:%s' % hyperparameters.task_type)


def train_epoch_classification(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args=None):
	loss_sum = 0.0

	model.train()

	batches_per_epoch = len(loader)

	epoch_start = time.time()

	n_samples = 0

	if hyperparameters.scheduler == 'cosine':
		lrs = cosine_scheduler(base_value=hyperparameters.optimizer['lr'], final_value=0.0, epochs=total_epochs, niter_per_ep=batches_per_epoch,
							   warmup_epochs=0, start_warmup_value=0, warmup_steps=-1)
	else:
		lr = hyperparameters.optimizer['lr']
		lrs = [lr for step in range(total_epochs*steps_per_epoch)]

	if 'use_autocast' not in args or args['use_autocast'] == False:
		print('not using autocast!')

	for i, batch in enumerate(loader):
		samples, targets = batch[0], batch[1]
		samples = samples.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)  # .view(-1)
		if len(batch) == 3: #fairness setup
			sensitive_attribute = batch[2].to(
				device, non_blocking=True).view(-1)

		lr = lrs[len(loader)*epoch+i]

		optimizer.zero_grad(set_to_none=True)

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		if hyperparameters.task_type != 'adv_classification':
			if args['use_autocast'] == False:
				output = model(samples)
				#print(output.shape, targets.shape, output.min(), output.max())
				if isinstance(criterion, SPLoss) or isinstance(criterion, CESPLoss):
					loss = criterion(output, targets, sensitive_attribute)
				else:
					loss = criterion(output, targets)
			else:
				with torch.cuda.amp.autocast():
					output = model(samples)
					#print(output.shape, targets.shape, output.min(), output.max())
					if isinstance(criterion, SPLoss) or isinstance(criterion, CESPLoss):
						loss = criterion(output, targets, sensitive_attribute)
					else:
						loss = criterion(output, targets)

		else:
			if args['use_autocast'] == False:
				loss = criterion(model=model, x_natural=samples,
								 target=targets, optimizer=optimizer)
			else:
				with torch.cuda.amp.autocast():
					loss = criterion(model=model, x_natural=samples,
									 target=targets, optimizer=optimizer)

		if args['use_autocast'] == False:
			loss.backward()
			optimizer.step()
		else:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

		loss_sum += loss.item() * samples.size(0)
		n_samples += samples.size(0)

	for param_group in optimizer.param_groups:
		lr = param_group['lr']

	print('epoch time', time.time()-epoch_start)

	return {
		'loss': loss_sum / n_samples,
		'score': 0.0,
		'lr': lr,
	}


def train_epoch_fairness(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args=None):
	loss_sum = 0.0

	model.train()

	batches_per_epoch = len(loader)

	epoch_start = time.time()

	n_samples = 0

	steps_per_epoch = len(loader)
	if hyperparameters.scheduler == 'cosine':
		lrs = cosine_scheduler(base_value=hyperparameters.optimizer['lr'], final_value=0.0, epochs=total_epochs, niter_per_ep=batches_per_epoch,
							   warmup_epochs=0, start_warmup_value=0, warmup_steps=-1)
	else:
		lr = hyperparameters.optimizer['lr']
		lrs = [lr for step in range(total_epochs*steps_per_epoch)]

	for i, b in enumerate(loader):
		samples_n, samples_c = None, None
		samples = b['data']
		if isinstance(samples, list):
			samples, samples_c = samples
			samples = samples.to(device, non_blocking=True)
			samples_c = samples_c.to(device, non_blocking=True)
			#print(samples.shape, samples_c.shape)
		else:
			samples = samples.to(device, non_blocking=True)

		targets = b['labels'].long().view(-1).to(device, non_blocking=True)
		sensitive_attribute = b['sensitive_attribute'].to(
			device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		lr = lrs[steps_per_epoch*epoch+i]
		# print(f'{lr=}')
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		with torch.cuda.amp.autocast():
			if samples is not None and samples_c is not None:
				output = model(x_num=samples, x_cat=samples_c)
			else:
				output = model(samples)

			if isinstance(criterion, SPLoss) or isinstance(criterion, CESPLoss):
				loss = criterion(output, targets, sensitive_attribute)
			else:
				loss = criterion(output, targets)

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		loss_sum += loss.item() * samples.size(0)
		n_samples += samples.size(0)

		del b
		del samples
		del targets

	for param_group in optimizer.param_groups:
		lr = param_group['lr']

	print('epoch time', time.time()-epoch_start)

	return {
		'loss': loss_sum / n_samples,
		'score': 0.0,
		'lr': lr
	}


def train_epoch_tabular(device, loader, model, criterion, optimizer, scaler, hyperparameters, epoch, total_epochs, epochs_step, args=None):
	loss_sum = 0.0

	model.train()

	batches_per_epoch = len(loader)

	epoch_start = time.time()

	n_samples = 0

	steps_per_epoch = len(loader)
	if hyperparameters.scheduler == 'cosine':
		lrs = cosine_scheduler(base_value=hyperparameters.optimizer['lr'], final_value=0.0, epochs=total_epochs, niter_per_ep=batches_per_epoch,
							   warmup_epochs=0, start_warmup_value=0, warmup_steps=-1)
	else:
		lr = hyperparameters.optimizer['lr']
		lrs = [lr for step in range(total_epochs*steps_per_epoch)]

	for i, (samples, targets) in enumerate(loader):
		samples_n, samples_c = None, None
		if isinstance(samples, list):
			samples, samples_c = samples
			samples = samples.to(device, non_blocking=True)
			samples_c = samples_c.to(device, non_blocking=True)
		else:
			samples = samples.to(device, non_blocking=True)

		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)

		lr = lrs[steps_per_epoch*epoch+i]
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		with torch.cuda.amp.autocast():
			if samples is not None and samples_c is not None:
				output = model(x_num=samples, x_cat=samples_c)
			else:
				output = model(samples)

			loss = criterion(output, targets)

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		loss_sum += loss.item() * samples.size(0)
		n_samples += samples.size(0)

	for param_group in optimizer.param_groups:
		lr = param_group['lr']

	print('epoch time', time.time()-epoch_start)

	return {
		'loss': loss_sum / n_samples,
		'score': 0.0,
		'lr': lr
	}


def evaluate_adv_classification(device, loader, model, hyperparameters, criterion, args):
	pass

	model.eval()
	robust_err_total = 0.0
	natural_err_total = 0.0
	n = 0
	if 'use_autocast' not in args or args['use_autocast'] == False:
		use_autocast = False
	else:
		use_autocast = True

	for i, (input, target) in enumerate(loader):

		input = input.to(device)
		target = target.to(device).view(-1)

		batch_size = len(target)

		X, y = Variable(input, requires_grad=True), Variable(target)
		err_natural, err_robust = pgd_whitebox(
			model, X, y, use_autocast, num_steps=args['metric']['attack_steps'])
		robust_err_total += err_robust
		natural_err_total += err_natural
		n += batch_size

	natural_err_total = 1.0 - natural_err_total / n
	robust_err_total = 1.0 - robust_err_total / n

	model.train()

	result = {
		'loss': 0.0,
		'score': (natural_err_total, robust_err_total),
	}

	print(result)
	return result


def evaluate_fairness(device, loader, model, hyperparameters, criterion, args):
	pass

	model.eval()
	all_pred, all_gt, all_correct, all_raw_pred = [], [], [], []
	y_pred, y_true, s_all = [], [], []
	losses = []

	torch.nn.BCEWithLogitsLoss()
	fairness_criterions = []
	for name in list(args['metric']):
		if name != 'name1':
			metric_name = args['metric'][name]
			if metric_name == 'DEO':
				criterion = DEO()
			elif metric_name == 'DSP':
				criterion = DSP()
			else:
				raise ValueError('fairness metric_name should be one of [\'DEO\', \'DSP\'], found:%s' % metric_name)

			fairness_criterions.append(criterion)

	n = 0.0

	for i, b in enumerate(loader):

		samples_n, samples_c = None, None
		samples = b['data']
		if isinstance(samples, list):
			samples, samples_c = samples
			samples = samples.to(device, non_blocking=True)
			samples_c = samples_c.to(device, non_blocking=True)
			#print(samples.shape, samples_c.shape)
		else:
			samples = samples.to(device, non_blocking=True)

		targets = b['labels'].long().view(-1).to(device, non_blocking=True)
		sensitive_attribute = b['sensitive_attribute'].to(
			device, non_blocking=True)

		with torch.cuda.amp.autocast(), torch.no_grad():
			if samples is not None and samples_c is not None:
				output = model(x_num=samples, x_cat=samples_c).detach()
			else:
				output = model(samples).detach()

			output = torch.softmax(output, dim=1)
			output = output[:, 1]

			n += output.shape[0]

			if len(y_pred) == 0:
				y_pred = output.detach().cpu()
				y_true = targets.detach().cpu()
				s_all = sensitive_attribute.detach().cpu()
			else:
				y_pred = torch.cat(
					[y_pred.detach().cpu(), output.detach().cpu()])
				y_true = torch.cat(
					[y_true.detach().cpu(), targets.detach().cpu()])
				s_all = torch.cat(
					[s_all.detach().cpu(), sensitive_attribute.detach().cpu()])

	y_pred = np.array(y_pred.view(-1).detach().cpu())
	y_true = np.array(y_true.view(-1).detach().cpu())
	s_all = np.array(s_all.view(-1).detach().cpu())

	y_true = (y_true > 0.5).astype(np.int32)

	fairness_scores = []
	for criterion in fairness_criterions:
		metric = 1.0-criterion(preds=y_pred, labels=y_true,
							   sensitive_attribute=s_all)
		fairness_scores.append(metric)

	if args['metric']['name1'] == 'acc':
		y_pred = (y_pred > 0.5).astype(np.int32)
		metric1 = accuracy_score(y_true, y_pred)
	elif args['metric']['name1'] == 'ap':
		metric1 = metrics.average_precision_score(y_true, y_pred)
	else:
		raise ValueError('args[\'metric\'][\'name1\'] type should be one of [\'acc\', \'ap\'], found:%s' % args['metric']['name1'])

	losses = np.array(losses)

	model.train()
	return {
		'loss': losses.mean(),
		'score': tuple([metric1] + fairness_scores),
		'preds': np.array([]),
		'raw_preds': np.array([]),
		'gt': np.array([]),
		'preds_correct': np.array([])
	}


def evaluate_tabular_classification(device, loader, model, hyperparameters, criterion, args):
	pass

	model.eval()
	all_pred, all_gt, all_correct, all_raw_pred = [], [], [], []
	y_pred, y_true, s_all = [], [], []
	losses = []

	n = 0.0

	for i, (samples, targets) in enumerate(loader):

		samples_n, samples_c = None, None
		if isinstance(samples, list):
			samples, samples_c = samples
			samples = samples.to(device, non_blocking=True)
			samples_c = samples_c.to(device, non_blocking=True)
		else:
			samples = samples.to(device, non_blocking=True)

		targets = targets.to(device, non_blocking=True)

		with torch.cuda.amp.autocast(), torch.no_grad():
			if samples is not None and samples_c is not None:
				output = model(x_num=samples, x_cat=samples_c)
			else:
				output = model(samples)

			output = torch.softmax(output, dim=1)
			output = output[:, 1]

			n += output.shape[0]

			if len(y_pred) == 0:
				y_pred = output.detach().cpu()
				y_true = targets.detach().cpu()
			else:
				y_pred = torch.cat(
					[y_pred.detach().cpu(), output.detach().cpu()])
				y_true = torch.cat(
					[y_true.detach().cpu(), targets.detach().cpu()])

	y_pred = y_pred.view(-1).detach().cpu()
	y_true = y_true.view(-1).detach().cpu()

	if args['metric']['name1'] == 'precision' and args['metric']['name2'] == 'recall':
		y_pred = (np.array(y_pred) > 0.5).astype(np.int32)
		y_true = (np.array(y_true) > 0.5).astype(np.int32)
		metric1 = precision_score(y_true, y_pred)
		metric2 = recall_score(y_true, y_pred)

	else:
		raise ValueError('loss[\'name1\'] should be one of [\'precision\', \'recall\'], found:%s '% args['metric']['name1'])


	losses = np.array(losses)

	model.train()

	return {
		'loss': losses.mean(),
		'score': (metric1, metric2),
		'preds': np.array([]),
	}
