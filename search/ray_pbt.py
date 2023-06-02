# Code contains adapted functions from Ray Tune: https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pbt.py
import numpy as np
import os
import copy
from copy import deepcopy
import shutil
import torch
import utils

from data_providers import get_data_provider
from search.fitness_functions import HyperparametersSearch
from search.hyperparameters import Hyperparameters
import json
from search.mo_utils import *
import math

from ray.tune.schedulers import PopulationBasedTraining
from ray.air import session
from ray import tune
from ray.tune.experiment import Trial
from ray.tune.schedulers import TrialScheduler

from typing import Callable, Dict, List, Optional, Tuple
import logging
from ray.air._internal.checkpoint_manager import CheckpointStorage
from ray.tune import TuneError
from ray.tune.search import SearchGenerator
from ray.tune.schedulers import TrialScheduler
from ray.tune.search.variant_generator import format_vars
import random
from datetime import datetime
import ray

from utils import *

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["FUNCTION_SIZE_ERROR_THRESHOLD"] = "500"

def init_run(args, zero_init=False):
    utils.set_random_seeds(args.seed)
    data_provider = get_data_provider(**vars(args))
    function_create_model = get_function_create_model(args, data_provider)
    hyperparameters = Hyperparameters(
        **vars(args), num_classes=data_provider.n_classes)
    alphabet_array = np.array(hyperparameters.alphabet)
    base_folder = args.folder
    utils.createAndCleanFolder(base_folder + '/models')
    return alphabet_array, base_folder, data_provider, function_create_model, hyperparameters

logger = logging.getLogger(__name__)

def _make_experiment_tag(orig_tag: str, config: Dict, mutations: Dict) -> str:
    """Appends perturbed params to the trial name to show in the console."""

    resolved_vars = {}
    for k in mutations.keys():
        resolved_vars[("config", k)] = config[k]
    return "{}@perturbed[{}]".format(orig_tag, format_vars(resolved_vars))


def randomly_init_population_hparams(hyperparameters, alphabet_array, args):
    shuffled_arrays = [[] for k in range(len(alphabet_array))]
    for k in range(len(alphabet_array)):
        for p in range(args.population_size):
            shuffled_arrays[k] += list(np.random.permutation(alphabet_array[k]))

    population = []
    for i in range(args.population_size):
        hparams = []
        for k in range(len(alphabet_array)):
            hparams.append(shuffled_arrays[k][i])
        population.append(np.copy(hparams))

    return population


def _filter_mutated_params_from_config(
        config: Dict, hyperparam_mutations: Dict
) -> Dict:
    """Filter out hyperparameters from a config so that only parameters specified
    within hyperparam_mutations remain. This recursively filters nested configs.
    Example:
    >>> config = {
    ...     "a": {"b": 2, "c": 0, "d": {"e": 0.1}},
    ...     "f": {"g": 0.5},
    ... }
    >>> hyperparam_mutations = {
    ...     "a": {"b": [1, 2], "c": [-1, 0]},
    ... }
    >>> _filter_mutated_params_from_config(config, hyperparam_mutations) == {
    ...     "a": {"b": 2, "c": 0}
    ... }
    True
    Args:
            config: The config dict that we want to filter.
            hyperparam_mutations: A dict containing a subset of hyperparameters from
                    config, used to filter the config.
    Returns:
            mutated_params: A copy of config containing only params specified in
                    hyperparam_mutations
    """
    mutated_params = {}
    for param_name in config:
        if param_name not in hyperparam_mutations:
            continue

        if isinstance(config[param_name], dict):
            nested_params = _filter_mutated_params_from_config(
                config[param_name], hyperparam_mutations[param_name]
            )
            mutated_params[param_name] = nested_params
        else:
            mutated_params[param_name] = config[param_name]
    return mutated_params


def copy_weights(src_model_id, dst_model_id, folder, src_last_epoch, dst_last_epoch):
    src_model_files = os.listdir('%s/models' % folder)
    src_model_files = [model_file for model_file in src_model_files if int(
        model_file.split('_')[1]) == src_model_id]

    dst_model_files = os.listdir('%s/models' % folder)
    dst_model_files = [model_file for model_file in dst_model_files if int(
        model_file.split('_')[1]) == dst_model_id]

    if len(src_model_files):
        src_file = [x for x in src_model_files if int(
            x.split('_')[2]) == src_last_epoch][0]
        dst_file = [x for x in dst_model_files if int(
            x.split('_')[2]) == dst_last_epoch][0]

        src_file_full_path = '%s/models/%s' % (folder, src_file)
        dst_file_full_path = '%s/models/%s' % (folder, dst_file)

        print('copy', src_file, '->', dst_file)
        shutil.copyfile(src_file_full_path, dst_file_full_path)


def _explore(
    config: Dict,
    mutations: Dict,
    resample_probability: float,
    perturbation_factors: Tuple[float],
    custom_explore_fn: Optional[Callable],
    random_seed: int,
    mutation_type='PBA'
) -> Tuple[Dict, Dict]:
    """Return a perturbed config and string descriptors of the operations performed
    on the original config to produce the new config.
    Args:
            config: Original hyperparameter configuration.
            mutations: Specification of mutations to perform as documented
                    in the PopulationBasedTraining scheduler.
            resample_probability: Probability of allowing resampling of a
                    particular variable.
            perturbation_factors: Scaling factors to choose between when mutating
                    a continuous hyperparameter.
            custom_explore_fn: Custom explore function applied after built-in
                    config perturbations.
    Returns:
            new_config: New hyperparameter configuration (after random mutations).
            operations: Map of hyperparams -> strings describing mutation operations
                    performed
    """

    operations = {}
    new_config = copy.deepcopy(config)
    for key, distribution in mutations.items():
        if key == 'model_id':
            continue

        if isinstance(distribution, (list, tuple)):
            # Case 1: Hyperparameter resample distribution is a list/tuple
            if mutation_type == 'PBA':
                if (
                        random.random() < resample_probability
                        or config[key] not in distribution
                ):
                    # Resample a value from the list with `resample_probability`
                    new_config[key] = random.choice(distribution)
                    operations[key] = "resample"
                else:
                    # Otherwise, perturb by shifting to the left or right of the list
                    shift = random.choice([0, 1, 2, 3]) * \
                        random.choice([-1, 1])
                    old_idx = distribution.index(config[key])
                    new_idx = old_idx + shift
                    new_idx = min(max(new_idx, 0), len(distribution) - 1)
                    new_config[key] = distribution[new_idx]
                    operations[key] = (
                        f"shift {shift}"
                        f"{' (noop)' if old_idx == new_idx else ''}"
                    )
            elif mutation_type == 'random':
                resample_probability = 1.0/len(mutations)
                if (
                        random.random() < resample_probability
                        or config[key] not in distribution
                ):
                    # Resample a value from the list with `resample_probability`
                    new_config[key] = random.choice(distribution)
                    operations[key] = "resample"
                else:
                    pass

        else:
            raise ValueError(
                f"Unsupported hyperparameter distribution type: {type(distribution)}"
            )
    if custom_explore_fn:
        # The user can perform any additional hyperparameter exploration
        # via `custom_explore_fn`
        new_config = custom_explore_fn(new_config)
        assert new_config is not None, "Custom explore fn failed to return new config"

    return new_config, operations


class _PBTTrialState:
    """Internal PBT state tracked per-trial."""

    def __init__(self, trial: Trial):
        self.orig_tag = trial.experiment_tag
        self.last_score = None
        self.last_checkpoint = None
        self.last_perturbation_time = 0
        self.last_train_time = 0  # Used for synchronous mode.
        self.last_result = None  # Used for synchronous mode.
        self.max_finished_epoch = None
        self.collected_fitness_values = []

    def __repr__(self) -> str:
        return str(
            (
                self.last_score,
                self.last_checkpoint,
                self.last_train_time,
                self.last_perturbation_time,
            )
        )


class MOPBT(PopulationBasedTraining):
    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        if trial_runner.search_alg is not None and isinstance(
                trial_runner.search_alg, SearchGenerator
        ):
            raise ValueError(
                "Search algorithms cannot be used with {} "
                "schedulers. Please remove {}.".format(
                    self.__class__.__name__, trial_runner.search_alg
                )
            )

        if not self._metric or not self._metric_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.TuneConfig()`".format(
                    self.__class__.__name__, self._metric, self._mode
                )
            )

        self._trial_state[trial] = _PBTTrialState(trial)

        for attr in self._hyperparam_mutations.keys():
            # Add attr to trial's config by sampling search space from
            # hyperparam_mutations.
            i = int(trial.trial_id.split('_')[-1])
            j = int(attr.replace('x', ''))
            trial.config[attr] = self.init_population[i][j]
            #self._fill_config(trial.config, attr, self._hyperparam_mutations[attr])
            # Make sure this attribute is added to CLI output.
            trial.evaluated_params[attr] = trial.config[attr]
        
    def _save_trial_state(
            self, state: _PBTTrialState, time: int, result: Dict, trial: Trial
    ):
        """Saves necessary trial information when result is received.
        Args:
                state: The state object for the trial.
                time: The current timestep of the trial.
                result: The trial's result dictionary.
                trial: The trial object.
        """

        # This trial has reached its perturbation interval.
        # Record new state in the state object.
        score = self._metric_op * result[self._metric]
        state.last_score = score
        state.last_train_time = time
        state.last_result = result

        state.collected_fitness_values.append(score)

        model_id = int(trial.trial_id.split('_')[-1])

        max_finished_epoch = 0
        files = os.listdir('%s/models/' % self.base_folder)
        for file in files:
            model_id_, epoch = int(file.split('_')[1]), int(file.split('_')[2])
            if model_id_ != model_id:
                continue
            if epoch > max_finished_epoch:
                max_finished_epoch = epoch

        state.max_finished_epoch = max_finished_epoch

    def on_trial_result(
            self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        if self._time_attr not in result:
            time_missing_msg = (
                "Cannot find time_attr {} "
                "in trial result {}. Make sure that this "
                "attribute is returned in the "
                "results of your Trainable.".format(
                    self._time_attr, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    time_missing_msg
                    + "If this error is expected, you can change this to "
                    "a warning message by "
                    "setting PBT(require_attrs=False)"
                )
            else:
                if log_once("pbt-time_attr-error"):
                    logger.warning(time_missing_msg)
        if self._metric not in result:
            metric_missing_msg = (
                "Cannot find metric {} in trial result {}. "
                "Make sure that this attribute is returned "
                "in the "
                "results of your Trainable.".format(self._metric, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    metric_missing_msg + "If this error is expected, "
                    "you can change this to a warning message by "
                    "setting PBT(require_attrs=False)"
                )
            else:
                if log_once("pbt-metric-error"):
                    logger.warning(metric_missing_msg)

        if self._metric not in result or self._time_attr not in result:
            return TrialScheduler.CONTINUE

        time = result[self._time_attr]
        state = self._trial_state[trial]

        # Continue training if burn-in period has not been reached, yet.
        if time < self._burn_in_period:
            return TrialScheduler.CONTINUE

        # Continue training if perturbation interval has not been reached, yet.
        if time - state.last_perturbation_time < self._perturbation_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead

        self._save_trial_state(state, time, result, trial)
        random_seed = self.random_seed*1234 + \
            int(trial.trial_id.split('_')[-1])*5678 + state.max_finished_epoch

        if not self._synch:
            state.last_perturbation_time = time
            lower_quantile, upper_quantile = self._quantiles()
            decision = TrialScheduler.CONTINUE
            for other_trial in trial_runner.get_trials():
                if other_trial.status in [Trial.PENDING, Trial.PAUSED]:
                    decision = TrialScheduler.PAUSE
                    break
            self._checkpoint_or_exploit(
                trial, trial_runner, upper_quantile, lower_quantile, random_seed
            )
            return TrialScheduler.NOOP if trial.status == Trial.PAUSED else decision
        else:
            # Synchronous mode.
            if any(
                    self._trial_state[t].last_train_time < self._next_perturbation_sync
                    and t != trial
                    for t in trial_runner.get_live_trials()
            ):
                logger.debug("Pausing trial {}".format(trial))
            else:
                # All trials are synced at the same timestep.
                lower_quantile, upper_quantile = self._quantiles()
                all_trials = trial_runner.get_trials()
                not_in_quantile = []
                for t in all_trials:
                    if t not in lower_quantile and t not in upper_quantile:
                        not_in_quantile.append(t)
                # Move upper quantile trials to beginning and lower quantile
                # to end. This ensures that checkpointing of strong trials
                # occurs before exploiting of weaker ones.
                all_trials = upper_quantile + not_in_quantile + lower_quantile
                for t in all_trials:
                    logger.debug("Perturbing Trial {}".format(t))
                    self._trial_state[t].last_perturbation_time = time
                    self._checkpoint_or_exploit(
                        t, trial_runner, upper_quantile, lower_quantile, random_seed
                    )

                all_train_times = [
                    self._trial_state[t].last_train_time
                    for t in trial_runner.get_trials()
                ]
                max_last_train_time = max(all_train_times)
                self._next_perturbation_sync = max(
                    self._next_perturbation_sync + self._perturbation_interval,
                    max_last_train_time,
                )
            # In sync mode we should pause all trials once result comes in.
            # Once a perturbation step happens for all trials, they should
            # still all be paused.
            # choose_trial_to_run will then pick the next trial to run out of
            # the paused trials.
            return (
                TrialScheduler.NOOP
                if trial.status == Trial.PAUSED
                else TrialScheduler.PAUSE
            )

    def _checkpoint_or_exploit(
            self,
            trial: Trial,
            trial_runner: "trial_runner.TrialRunner",
            upper_quantile: List[Trial],
            lower_quantile: List[Trial],
            random_seed: int
    ):
        if self.random_search:
            return

        """Checkpoint if in upper quantile, exploits if in lower."""
        trial_executor = trial_runner.trial_executor
        state = self._trial_state[trial]
        if trial in upper_quantile:
            # The trial last result is only updated after the scheduler
            # callback. So, we override with the current result.
            logger.debug("Trial {} is in upper quantile".format(trial))
            logger.debug("Checkpointing {}".format(trial))
            if trial.status == Trial.PAUSED:
                # Paused trial will always have an in-memory checkpoint.
                state.last_checkpoint = trial.checkpoint
            else:
                state.last_checkpoint = trial_executor.save(
                    trial, CheckpointStorage.MEMORY, result=state.last_result
                )
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            logger.debug("Trial {} is in lower quantile".format(trial))

            trial_to_clone = random.choice(upper_quantile)
            assert trial is not trial_to_clone
            if not self._trial_state[trial_to_clone].last_checkpoint:
                logger.info(
                    "[pbt]: no checkpoint for trial."
                    " Skip exploit for Trial {}".format(trial)
                )
                print(
                    " Skip exploit for Trial {}".format(trial)
                )

                return
            self._exploit(trial_runner, trial, trial_to_clone, random_seed)

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial, random_seed: int) -> Tuple[Dict, Dict]:
        """Gets new config for trial by exploring trial_to_clone's config.
        Args:
                trial: The current trial that decided to exploit trial_to_clone.
                trial_to_clone: The top-performing trial with a hyperparameter config
                        that the current trial will explore by perturbing.
        Returns:
                new_config: New hyperparameter configuration (after random mutations).
                operations: Map of hyperparams -> strings describing mutation operations
                        performed
        """
        return _explore(
            trial_to_clone.config,
            self._hyperparam_mutations,
            self._resample_probability,
            self._perturbation_factors,
            self._custom_explore_fn,
            random_seed,
            self.mutation_type
        )

    def _exploit(
            self,
            trial_runner: "trial_runner.TrialRunner",
            trial: Trial,
            trial_to_clone: Trial,
            random_seed: int
    ):
        """Transfers perturbed state from trial_to_clone -> trial.
        If specified, also logs the updated hyperparam state.
        """
        trial_state = self._trial_state[trial]
        new_state = self._trial_state[trial_to_clone]
        class_name = self.__class__.__name__
        # logger.info(
        # 	f"\n\n[{class_name}] [Exploit] Cloning trial "
        # 	"{} (score = {:4f}) into trial {} (score = {:4f})\n".format(
        # 		trial_to_clone.trial_id,
        # 		new_state.last_score,
        # 		trial.trial_id,
        # 		trial_state.last_score,
        # 	)
        # )

        new_config, operations = self._get_new_config(
            trial, trial_to_clone, random_seed)

        # Only log mutated hyperparameters and not entire config.
        old_params = _filter_mutated_params_from_config(
            trial_to_clone.config, self._hyperparam_mutations
        )
        new_params = _filter_mutated_params_from_config(
            new_config, self._hyperparam_mutations
        )
        explore_info_str = (
            f"\n\n[{class_name}] [Explore] Perturbed the hyperparameter config of trial"
            f"{trial.trial_id}:\n"
        )
        explore_info_str += (
            self._summarize_hyperparam_changes(
                old_params, new_params, operations)
            or "No hyperparameters mutated."
        )
        logger.info(explore_info_str)

        if self._log_config:
            self._log_config_on_step(
                trial_state, new_state, trial, trial_to_clone, new_config
            )

        new_tag = _make_experiment_tag(
            trial_state.orig_tag, new_config, self._hyperparam_mutations
        )
        if trial.status == Trial.PAUSED:
            # If trial is paused we update it with a new checkpoint.
            # When the trial is started again, the new checkpoint is used.
            if not self._synch:
                raise TuneError(
                    "Trials should be paused here only if in "
                    "synchronous mode. If you encounter this error"
                    " please raise an issue on Ray Github."
                )
        else:
            trial_runner.pause_trial(trial, should_checkpoint=False)
        trial.set_experiment_tag(new_tag)
        # Clone hyperparameters from the `trial_to_clone`
        cur_trial_model_id = int(trial.trial_id.split('_')[-1])
        trial_model_id_to_clone = int(trial_to_clone.trial_id.split('_')[-1])

        print('weights copy', trial_model_id_to_clone, '->', cur_trial_model_id)
        copy_weights(trial_model_id_to_clone, cur_trial_model_id, self.base_folder,
                     new_state.max_finished_epoch, trial_state.max_finished_epoch)

        trial.set_config(new_config)
        # Resume training from a shallow copy of `trial_to_clone`'s latest checkpoint
        checkpoint_to_exploit = copy.copy(new_state.last_checkpoint)
        # NOTE: Clear the checkpoint id (which was set by the other trial's
        # checkpoint manager) so that the current trial's checkpoint manager marks
        # the checkpoint as the most recent to use upon trial resume
        checkpoint_to_exploit.id = None
        trial.on_checkpoint(checkpoint_to_exploit)
        self._num_perturbations += 1
        # Transfer over the last perturbation time as well
        trial_state.last_perturbation_time = new_state.last_perturbation_time
        trial_state.last_train_time = new_state.last_train_time
        # exit(0)
        print("new config", trial_to_clone.config, '->',
              trial.config, trial.trial_id, operations)

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in the lower and upper `quantile` of the population.
        If there is not enough data to compute this, returns empty lists.
        """
        trials = []
        for trial, state in self._trial_state.items():
            logger.debug("Trial {}, state {}".format(trial, state))
            if trial.is_finished():
                logger.debug("Trial {} is finished".format(trial))
            if state.last_score is not None and not trial.is_finished():
                trials.append(trial)
        
        fitness_values = [self._trial_state[t].last_score for t in trials]
        fitness_values_archive = []
        for t in trials:
            fitness_values_archive += self._trial_state[t].collected_fitness_values
        fitness_values = assign_actual_fitness(fitness_values, self.objective, self.n_objectives,
                                               self.constraints, fitness_values_archive, os.path.join(self.base_folder, 'results'))
        sort_ind = my_argsort(fitness_values)

        #trials.sort(key=lambda t: self._trial_state[t].last_score)
        trials_ = [trials[i] for i in sort_ind]
        trials = trials_
        print('after sort', [
              (t.trial_id, self._trial_state[t].last_score) for t in trials])

        if len(trials) <= 1:
            return [], []
        else:
            num_trials_in_quantile = int(
                math.ceil(len(trials) * self._quantile_fraction)
            )
            if num_trials_in_quantile > len(trials) / 2:
                num_trials_in_quantile = int(math.floor(len(trials) / 2))
            return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])

    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner"
    ) -> Optional[Trial]:
        """Ensures all trials get fair share of time (as defined by time_attr).
        This enables the PBT scheduler to support a greater number of
        concurrent trials than can fit in the cluster at any given time.
        """
        candidates = []
        for trial in trial_runner.get_trials():
            if trial.status in [
                    Trial.PENDING,
                    Trial.PAUSED,
            ] and trial_runner.trial_executor.has_resources_for_trial(trial):
                if not self._synch:
                    candidates.append(trial)
                elif (
                        self._trial_state[trial].last_train_time
                        < self._next_perturbation_sync
                ):
                    candidates.append(trial)
        candidates.sort(
            key=lambda trial: self._trial_state[trial].last_train_time)
        return candidates[0] if candidates else None


def mytrain(config, args, MAX_EPOCHS, alphabet_array, base_folder, data_provider, function_create_model, hyperparameters, checkpoint_dir=None):
    EPOCHS_STEP = MAX_EPOCHS // int(args.steps)
    MAX_EPOCHS//EPOCHS_STEP

    fitness_function = HyperparametersSearch(base_folder, MAX_EPOCHS, EPOCHS_STEP, None, hyperparameters, function_create_model)

    model_id = int(session.get_trial_id().split('_')[-1])

    prev_epoch = 0
    files = os.listdir('%s/models/' % base_folder)

    files_ = []
    for file in files:
        model_id_, epoch = int(file.split('_')[1]), int(file.split('_')[2])
        if model_id_ != model_id:
            continue
        files_.append(file)
        if epoch > prev_epoch:
            prev_epoch = epoch

    prev_step = prev_epoch // EPOCHS_STEP
    last_step = args.steps
    for step in range(prev_step, last_step):

        cur_epoch = step*EPOCHS_STEP

        train_seed = 1234 * args.seed + 5678 * cur_epoch + model_id
        weights_seed = 1234 * args.seed + model_id

        x = []
        for j in range(len(alphabet_array)):
            x.append(config['x%d' % j])
            # print(x)
        print('train setup: config={} | model_id={} | cur_epoch={} | weights_seed={} | training_seed={}'.format(config, model_id,
              cur_epoch, weights_seed, train_seed))
        val_score, test_score, *_ = fitness_function.fitness(
            x, model_id, cur_epoch, weights_seed=weights_seed, train_seed=train_seed, cur_data_provider=data_provider)
        fitness = np.array(val_score)

        torch.cuda.empty_cache()
        if not ('keep_model_files' in vars(args) and args.keep_model_files):
            utils.delete_old_model_files(base_folder, model_id, cur_epoch)

        config_ = deepcopy(config)
        for x in config_:
            config_[x] = int(config[x])

        filename = os.path.join(base_folder, 'results', '%d.json' % model_id)
        if not os.path.exists(filename):
            cur_results = {}
        else:
            cur_results = json.load(open(filename, 'r'))
        cur_results[cur_epoch+EPOCHS_STEP] = {'val_score': val_score, 'test_score': test_score,
                                              'time': datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                                              'config': dict(config_)
                                              }
        json.dump(cur_results, open(filename, 'w'))

        config['model_id'] = model_id
        session.report({"mo_score": np.array(fitness),
                       'training_iteration': step+1})


def runRayPBT(args):

    alphabet_array, base_folder,  data_provider, function_create_model, hyperparameters = init_run(
        args)
    MAX_EPOCHS = args.max_epochs
    EPOCHS_STEP = MAX_EPOCHS // int(args.steps)
    MAX_EPOCHS//EPOCHS_STEP

    ray.init(_temp_dir='/export/scratch1/home/arkadiy/ray_results')

    search_space = {}
    search_space_with_lists = {}

    for j in range(len(alphabet_array)):
        options = np.arange(alphabet_array[j])
        search_space['x%d' % j] = tune.choice(list(options))
        search_space_with_lists['x%d' % j] = list(options)

    population = randomly_init_population_hparams(
        hyperparameters, alphabet_array, args)

    quantile_fraction = 0.25
    if 'selection' in vars(args):
        quantile_fraction = float(args.selection)

    scheduler = MOPBT(
        time_attr="training_iteration",
        metric="mo_score",
        mode="max",
        perturbation_interval=1,
        resample_probability=0.2,
        quantile_fraction=quantile_fraction,
        hyperparam_mutations=search_space_with_lists,
        perturbation_factors=(0.5, 1.5),
    )
    scheduler.base_folder = base_folder
    scheduler.objective = args.search
    scheduler.init_population = population
    scheduler._synch = args.sync
    scheduler.random_search = args.search == 'RandomSearch'
    scheduler.random_seed = args.seed
    scheduler.mutation_type = args.mutation
    scheduler.n_objectives = max(len(args.metric), 2)
    if 'constraints' in vars(args):
        scheduler.constraints = args.constraints
    else:
        scheduler.constraints = None
    print(f'{scheduler.n_objectives=}')
    print(f'{scheduler.mutation_type=}')
    print(f'{quantile_fraction=}')
    print(f'{scheduler.constraints=}')

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(mytrain, args=args, MAX_EPOCHS=MAX_EPOCHS, alphabet_array=alphabet_array, base_folder=base_folder,
                                 data_provider=data_provider, function_create_model=function_create_model, hyperparameters=hyperparameters),
            resources={"cpu": 1, "gpu": 1.0/args.parallel_workers_per_gpu}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.population_size,
        ),
        param_space=search_space,
    )

    utils.createAndCleanFolder(os.path.join(base_folder, 'results'))

    tuner.fit()

    ray.shutdown()

    try:
        shutil.rmtree('/tmp/ray')
        shutil.rmtree('/tmp/_ray_lockfiles')
    except Exception:
        pass
