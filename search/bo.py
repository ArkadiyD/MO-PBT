#this code is adapted from  https://botorch.org/tutorials/multi_objective_bo
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.multi_objective.scalarization import \
	get_chebyshev_scalarization
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gp_regression import FixedNoiseGP
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.monte_carlo import (
	qNoisyExpectedHypervolumeImprovement)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch import fit_gpytorch_mll
 
from search.mo_utils import *
from ray import tune
import json
import os
import shutil
from copy import deepcopy

import numpy as np
import ray
import torch
 
import utils
from data_providers import get_data_provider
from search.fitness_functions import *
from search.hyperparameters import Hyperparameters
from search.ray_pbt import *

def parallel_training(args, base_folder, alphabet_array, hyperparameters, function_create_model, data_provider, search_space, search_space_with_lists, population, epochs_step):
	print('population for parallel training:', population)
	scheduler = MOPBT(
		time_attr="training_iteration",
		metric="mo_score",
		mode="max",
		perturbation_interval=1,
		resample_probability=0.2,
		quantile_fraction=0.25,
		hyperparam_mutations=search_space_with_lists,
		perturbation_factors=(0.5,1.5),
		)
	scheduler.base_folder = base_folder
	scheduler.objective = args.search
	scheduler.init_population = population
	scheduler._synch = True
	scheduler.random_search = 'RandomSearch' 
	scheduler.objective = 'RandomSearch'
	scheduler.random_seed = args.seed
	scheduler.n_objectives = max(len(args.metric), 2)
	if 'constraints' in vars(args):
		scheduler.constraints = args.constraints
	else:
		scheduler.constraints = None
	print(f'{scheduler.n_objectives=}')
	print(f'{scheduler.constraints=}')
	print(f'{epochs_step=}')

	tuner = tune.Tuner(
		tune.with_resources(
			tune.with_parameters(mytrain, args=args, MAX_EPOCHS=args.max_epochs, alphabet_array=alphabet_array, base_folder=base_folder, 
			data_provider=data_provider, function_create_model=function_create_model, hyperparameters=hyperparameters),
			resources={"cpu": 1, "gpu": 1.0/args.parallel_workers_per_gpu}
		),
		tune_config=tune.TuneConfig(
			scheduler=scheduler,
			num_samples=len(population),
		),
		param_space=search_space,
	)

	utils.createAndCleanFolder(os.path.join(base_folder, 'results'))
	utils.createAndCleanFolder(os.path.join(base_folder, 'models'))

	tuner.fit() 


class MyObjective(MultiObjectiveTestProblem):
	num_objectives = 2
	_ref_point = [-0.1, -0.1]

	def __init__(self, alphabet_array, limits, base_folder, MAX_EPOCHS, EPOCHS_STEP, data_provider, hyperparameters, function_create_model, fitness_function, args):
		self.alphabet_array = alphabet_array
		self._bounds = []
		for k in range(len(alphabet_array)):
			self._bounds.append((limits[k][0]-0.5, limits[k][1]+0.5))
			
		self.search_space = {}
		self.search_space_with_lists = {}		
		for j in range(len(alphabet_array)):
			options = np.arange(alphabet_array[j])
			self.search_space['x%d'%j] = tune.choice(list(options))
			self.search_space_with_lists['x%d'%j] = list(options)
		
		print('bounds:', self._bounds)
		self.dim = len(alphabet_array)
		self.solutions = {}
		self.model_ids = 0
		self.base_folder = base_folder
		self.hyperparameters = hyperparameters
		self.MAX_EPOCHS, self.EPOCHS_STEP, self.data_provider = MAX_EPOCHS, EPOCHS_STEP, data_provider
		self.function_create_model = function_create_model
		self.data_provider = data_provider
		self.args = args
		self.solutions = {}
		self.write_cnt = 0
		self.fitness_function = fitness_function
		super().__init__()

	def evaluate_true(self, X):
		torch.cuda.is_available = lambda: True

		individual = {'model_id': 0, 'fitnesses': [], 'val_scores': [], 'test_scores': [], 'val_accs': [], 'test_accs': [], 'val_losses': [], 'test_losses': [], 'val_preds': [],
					  'train_scores': [], 'train_losses': [], 'val_raw_preds': [], 'val_gt': [], 'test_preds': [], 'test_gt': [], 'hyperparameters_history': [], 'op_params': []}

		print('evaluate', X)
		X_ = X.to("cpu")
		X_ = np.array(X_)

		X_ = np.round(X_, 0).astype(np.int32)
		for j in range(X_.shape[1]):
			X_[:, j] = np.clip(X_[:, j], self._bounds[j]
								[0]+0.5, self._bounds[j][1]-0.5)

		val_scores = []
		population = []
		for k in range(X_.shape[0]):
			x = tuple(np.array(X_[k]))
			cur_individual = deepcopy(individual)
			cur_individual['hyperparameters'] = x
			cur_individual['model_id'] = self.model_ids
			self.model_ids += 1
			population.append(cur_individual)

		population_for_train = [x['hyperparameters'] for x in population]
		parallel_training(self.args, self.base_folder, self.alphabet_array, self.hyperparameters, self.function_create_model, self.data_provider, self.search_space, self.search_space_with_lists, population_for_train, self.EPOCHS_STEP)
		##############
		for file in os.listdir(os.path.join(self.base_folder, 'results')):
			src_file_full_path = os.path.join(self.base_folder, 'results', file)
			cnt = len(os.listdir(os.path.join(self.base_folder, 'all_results')))
			dst_file_full_path = os.path.join(self.base_folder, 'all_results', '%d.json'%cnt)			
			shutil.copyfile(src_file_full_path, dst_file_full_path)

		val_scores = []
		for i in range(len(population)):
			data = json.load(open(os.path.join(self.base_folder, 'results', '%d.json'%i), 'r'))
			val_score = data[str(self.MAX_EPOCHS)]['val_score']
			val_scores.append(val_score)
			utils.delete_old_model_files(
				self.base_folder, population[i]['model_id'], self.MAX_EPOCHS+1)
		val_scores = np.array(val_scores)
		print('gathered_val_scores', val_scores)

		torch.cuda.is_available = lambda: False

		return torch.tensor(val_scores)

def get_bo_ref_point(folder, n_objectives, beta=0.1):
	files = os.listdir(folder)
	scores = []
	for file in files:
		data = json.load(open(os.path.join(folder, file), 'r'))
		for ep in data:
			scores.append(data[ep]['val_score'])
	scores = np.array(scores).reshape(-1,n_objectives)
	
	ref_point = []
	for k in range(n_objectives):
		max_obj = np.max(scores[:,k])
		min_obj = np.min(scores[:,k])
		
		b = min_obj - beta*(max_obj-min_obj)
		ref_point.append(b)
	
	ref_point = np.array(ref_point)
	print('ref point', ref_point)
	return ref_point

def runOptimizerBoTorch(args):
	torch.cuda.is_available = lambda: False

	N_BATCH = int(args.bo_batches)
	MC_SAMPLES = 128
	BATCH_SIZE = int(args.bo_batch_size)
	RAW_SAMPLES = 4
	NUM_RESTARTS = 2

	ray.init(_temp_dir='/export/scratch1/home/arkadiy/ray_results')
	

	alphabet_array, base_folder,  data_provider, function_create_model, hyperparameters = init_run(args)
	MAX_EPOCHS = args.max_epochs
	EPOCHS_STEP = MAX_EPOCHS // int(args.steps)
	print(MAX_EPOCHS, EPOCHS_STEP)

	tkwargs = {
		"dtype": torch.double,
		"device": torch.device("cpu"),
	}
	limits = [(0, alphabet_array[k]-1) for k in range(len(alphabet_array))]
	print('limits:',limits)

	fitness_function = HyperparametersSearch(base_folder, MAX_EPOCHS, EPOCHS_STEP, data_provider, hyperparameters, function_create_model)

	problem = MyObjective(alphabet_array, limits, base_folder, MAX_EPOCHS, EPOCHS_STEP,
						  data_provider, hyperparameters, function_create_model, fitness_function, args).to(**tkwargs)

	standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
	standard_bounds[1] = 1

	utils.createAndCleanFolder(os.path.join(base_folder, 'all_results'))

	def generate_initial_data(n=6):
		# generate training data
		train_x = draw_sobol_samples(
			bounds=problem.bounds, n=n, q=1).squeeze(1)
		print('train X:', train_x)
		train_obj = train_obj_true = problem(train_x)
		print('train obj:', train_obj)
		return train_x, train_obj, train_obj_true

	def initialize_model(train_x_, train_obj):
		# define models for objective and constraint
		train_x = normalize(train_x_, problem.bounds)
		print('normalized train X:', train_x)
		models = []
		for i in range(train_obj.shape[-1]):
			train_y = train_obj[..., i:i+1]
			# essentially noiseless setting
			train_yvar = torch.full_like(train_y, 1e-6)
			models.append(
				FixedNoiseGP(train_x, train_y, train_yvar,
							 outcome_transform=Standardize(m=1))
			)
		model = ModelListGP(*models)
		mll = SumMarginalLogLikelihood(model.likelihood, model)
		return mll, model

	def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
		"""Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
		# partition non-dominated space into disjoint rectangles
		acq_func = qNoisyExpectedHypervolumeImprovement(
			model=model,
			ref_point=problem.ref_point.tolist(),  # use known reference point 
			X_baseline=normalize(train_x, problem.bounds),
			prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
			sampler=sampler,
		)
		# optimize
		candidates, _ = optimize_acqf(
			acq_function=acq_func,
			bounds=standard_bounds,
			q=BATCH_SIZE,
			num_restarts=NUM_RESTARTS,
			raw_samples=RAW_SAMPLES,  # used for intialization heuristic
			options={"batch_limit": 5, "maxiter": 200},
			sequential=True,
		)
		# observe new values 
		print('candidates (normalized):', candidates)
		new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
		print('new_x:', new_x)
		new_obj = new_obj_true = problem(new_x)
		return new_x, new_obj, new_obj_true

	def optimize_qnparego_and_get_observation(model, train_x, train_obj, sampler):
		"""Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization 
		of the qNParEGO acquisition function, and returns a new candidate and observation."""
		"""Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization 
		of the qNParEGO acquisition function, and returns a new candidate and observation."""
		train_x = normalize(train_x, problem.bounds)
		with torch.no_grad():
			pred = model.posterior(train_x).mean
		acq_func_list = []
		for _ in range(BATCH_SIZE):
			weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
			objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
			acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
				model=model,
				objective=objective,
				X_baseline=train_x,
				sampler=sampler,
				prune_baseline=True,
			)
			acq_func_list.append(acq_func)
		# optimize
		candidates, _ = optimize_acqf_list(
			acq_function_list=acq_func_list,
			bounds=standard_bounds,
			num_restarts=NUM_RESTARTS,
			raw_samples=RAW_SAMPLES,  # used for intialization heuristic
			options={"batch_limit": 5, "maxiter": 200},
		)
		# observe new values 
		print('candidates (normalized):', candidates)
		new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
		print('new_x:', new_x)
		new_obj_true = problem(new_x)
		new_obj = new_obj_true
		return new_x, new_obj, new_obj_true
		
	def optimize_model_and_get_observation(args, model, train_x, train_obj, sampler):
		if args.search == 'qnehvi':
			return optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler)
		elif args.search == 'qnparego':
			return optimize_qnparego_and_get_observation(model, train_x, train_obj, sampler)
		else:
			raise ValueError('args.search for BO algorithms should be one of [\'qnehvi\', \'qnparego\'], found:%s ' % args.search)

	train_x, train_obj, train_obj_true = generate_initial_data(
		n=2*(problem.dim+1))
	ref_point = get_bo_ref_point(os.path.join(base_folder, 'all_results'), max(len(args.metric), 2))
	if args.search == 'qnehvi':
		problem._ref_point = problem.ref_point = torch.tensor(ref_point)

	mll, model = initialize_model(train_x, train_obj)

	for iteration in range(1, N_BATCH + 1):
		torch.cuda.is_available = lambda: False
		fit_gpytorch_mll(mll)
		
		# define the qEI and qNEI acquisition modules using a QMC sampler
		sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
		
		# optimize acquisition functions and get new observations
		new_x, new_obj, new_obj_true = optimize_model_and_get_observation(
			args, model, train_x, train_obj, sampler)
		
		train_x = torch.cat([train_x, new_x])
		train_obj = torch.cat([train_obj, new_obj])
		train_obj_true = torch.cat([train_obj_true, new_obj_true])

		mll, model = initialize_model(train_x, train_obj)
		
	ray.shutdown()

	