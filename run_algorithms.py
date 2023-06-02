import multiprocessing as mp
import datetime
from pathlib import Path
import argparse
import yaml
import torch
import shutil
from types import SimpleNamespace

from search.ray_pbt import *
from search.moasha import *
from search.bo import *

def run(config):
	args = SimpleNamespace(**config)

	log_folder = config['logs_path']
	Path(log_folder).mkdir(exist_ok=True)
	run_folder = args.out_name_template.format(**config)
	run_folder = os.path.join(log_folder, run_folder)
	Path(run_folder).mkdir(exist_ok=True)
	run_folder = os.path.join(run_folder, str(config['seed']))
	Path(run_folder).mkdir(exist_ok=True)
	# if the config file has been changed, the changed version will be copied, even though the config actually used will be the proper one
	shutil.copy(config_path, run_folder)
	args.folder = run_folder
	print(args)

	if args.algorithm == 'RandomSearch':
		runRandomSearch(args)

	elif args.algorithm == 'RayPBT':
		runRayPBT(args)

	elif args.algorithm == 'MOASHA':
		runMOASHA(args)
	elif args.algorithm == 'MOASHATPE':
		runMOASHA(args, tpesampler=True)

	elif args.algorithm == 'BoTorch':
		runOptimizerBoTorch(args)

	else:
		raise ValueError('args.algorithm should be one of [\'RandomSearch\', \'RayPBT\', \'MOASHA\', \'MOASHATPE\', \'BoTorch\'], found:%s ' % args.algorithm)

	if not ('keep_model_files' in vars(args) and args.keep_model_files):
		try:
			models_folder = os.path.join(run_folder, "models")
			shutil.rmtree(models_folder)
		except Exception as e:
			pass

	try:
		models_folder = os.path.join(run_folder, "full_results")
		shutil.rmtree(models_folder)
	except Exception as e:
		pass


def create_configs_all(config):
	configs_all = []
	for i_seed in range(config['n_seeds']):
		config_cur = deepcopy(config)
		config_cur['seed'] += i_seed
		config_cur['i_seed'] = i_seed
		configs_all.append(config_cur)
	for config in configs_all:
		print('seed:', config['seed'])

	return configs_all


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(f'--config', type=str)
	parsed_args = parser.parse_args()
	config_path = parsed_args.config
	config = yaml.safe_load(open(config_path))

	configs_all = create_configs_all(config)
	for config in configs_all:
		run(config)
