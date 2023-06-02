# Code contains adapted functions from Ray Tune: https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/async_hyperband.py
from ray.tune import ProgressReporter
from search.mo_utils import *
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.tune.experiment import Trial
from ray.tune.execution import trial_runner
from ray.air import session
from ray import air, tune
from typing import Dict, Optional, Union
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime

import numpy as np
import ray
import torch

import utils
from data_providers import get_data_provider
from search.fitness_functions import HyperparametersSearch
from search.hyperparameters import Hyperparameters
from search.tpesampler import *
from search.ray_pbt import *


class CustomReporter(ProgressReporter):

    def should_report(self, trials, done=False):
        return False

    def report(self, trials, *sys_info):
        print(*sys_info)
        print("\n".join([str(trial) for trial in trials]))


class MO_Bracket:
    """Bookkeeping system to track the cutoffs.
    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
    Example:
                    >>> trial1, trial2, trial3 = ... # doctest: +SKIP
                    >>> b = _Bracket(1, 10, 2, 0) # doctest: +SKIP
                    >>> # CONTINUE
                    >>> b.on_result(trial1, 1, 2) # doctest: +SKIP
                    >>> # CONTINUE
                    >>> b.on_result(trial2, 1, 4) # doctest: +SKIP
                    >>> # rungs are reversed
                    >>> b.cutoff(b._rungs[-1][1]) == 3.0 # doctest: +SKIP
                     # STOP
                    >>> b.on_result(trial3, 1, 1) # doctest: +SKIP
                    >>> b.cutoff(b._rungs[3][1]) == 2.0 # doctest: +SKIP
    """

    def __init__(
        self,
        min_t: int,
        max_t: int,
        reduction_factor: float,
        s: int,
        stop_last_trials: bool = True,
        selection='epsnet',
        base_folder='', args=None
    ):
        self.base_folder = base_folder
        self.args = args
        self.n_objectives = max(len(args.metric), 2)
        self.rf = reduction_factor

        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [
            (min_t * self.rf ** (k + s), {}) for k in reversed(range(MAX_RUNGS))
        ]
        self._stop_last_trials = stop_last_trials

        self.selection = selection

        if 'constraints' in vars(self.args):
            self.constraints = self.args.constraints
        else:
            self.constraints = None

    def selector_eps_net(self, solutions):  # solutions are tuple(x, trial_id)
        solutions = [{'fitnesses': [
            np.array(x[0])], 'trial_id':x[1]} for i, x in enumerate(list(solutions))]
        fronts = fastNonDominatedSorting(
            solutions, self.n_objectives, self.constraints)

        to_drop = int((1 - 1.0 / self.rf) * len(solutions))  # how many to drop
        to_select = len(solutions) - to_drop
        print('to_select {} solutions / {}'.format(to_select, len(solutions)))

        if to_select == 0:
            return []

        selected_solutions, not_selected_solutions = epsnet(
            fronts, to_select, self.n_objectives)
        assert len(selected_solutions) + \
            len(not_selected_solutions) == len(solutions)
        return selected_solutions

    def selector_hcbs_net_asmotpe(self, solutions):
        solutions = [{'fitnesses': [
            np.array(x[0])], 'trial_id':x[1]} for i, x in enumerate(list(solutions))]
        
        fronts = fastNonDominatedSorting(
            solutions, self.n_objectives, self.constraints)

        to_drop = int((1 - 1.0 / self.rf) * len(solutions))  # how many to drop
        to_select = len(solutions) - to_drop
        print('to_select {} solutions / {}'.format(to_select, len(solutions)))

        if to_select == 0:
            return []

        selected_solutions, not_selected_solutions = hcbs_oneref_asmotpe(
            fronts, to_select, self.n_objectives)
        assert len(selected_solutions) + \
            len(not_selected_solutions) == len(solutions)
        return selected_solutions

    def cutoff(self, recorded) -> Optional[Union[int, float, complex, np.ndarray]]:
        # real selection happens not here
        return 0.0

    def on_result(self, trial: Trial, cur_iter: int, cur_rew: Optional[float]) -> str:
        action = TrialScheduler.CONTINUE
        
        for milestone, recorded in self._rungs:
            #print(milestone, cur_iter, recorded)

            if (
                cur_iter >= milestone
                and trial.trial_id in recorded
                and not self._stop_last_trials
            ):
                # If our result has been recorded for this trial already, the
                # decision to continue training has already been made. Thus we can
                # skip new cutoff calculation and just continue training.
                # We can also break as milestones are descending.
                break
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                if cur_rew is not None:
                    recorded[trial.trial_id] = cur_rew
                else:
                    logger.warning(
                        "Reward attribute is None! Consider"
                        " reporting using a different field."
                    )

                if len(recorded) > 1:
                    recorded_values = [(recorded[x], x) for x in recorded]
                    if self.selection == 'epsnet':
                        selected_solutions = self.selector_eps_net(
                            recorded_values)
                    elif self.selection == 'HCBSMOTPE':
                        selected_solutions = self.selector_hcbs_net_asmotpe(
                            recorded_values)
                    else:
                        raise ValueError('selection type should be one of [\'epsnet\', \'HCBSMOTPE\'], found:%s' % self.selection)

                    selected_solutions_trial_ids = [
                        x['trial_id'] for x in selected_solutions]
                    print('selected_solutions_trial_ids:', selected_solutions_trial_ids)
                    if trial.trial_id not in selected_solutions_trial_ids:
                        action = TrialScheduler.STOP

                break
        return action

    def debug_str(self) -> str:
        # TODO: fix up the output for this
        iters = " | ".join(
            [
                "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
                for milestone, recorded in self._rungs
            ]
        )
        return "Bracket: " + iters


class MOASHA(AsyncHyperBandScheduler):
    def __init__(
        self,
        time_attr: str = "training_iteration",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 4,
        brackets: int = 1,
        stop_last_trials: bool = True,
        selection='epsnet',
        base_folder='', args=None,
                    TPESampler=False,
        search_space=None
    ):
        AsyncHyperBandScheduler.__init__(self)

        self.base_folder = base_folder
        self.args = args
        self.n_objectives = max(len(args.metric), 2)

        self.TPESampler = TPESampler
        self.search_space = search_space
        self.trial_configs = {}
        if 'constraints' in vars(self.args):
            self.constraints = self.args.constraints
        else:
            self.constraints = None

        self._brackets = [
            MO_Bracket(
                grace_period,
                max_t,
                reduction_factor,
                s,
                stop_last_trials=stop_last_trials,
                selection=selection,
                base_folder=self.base_folder,
                args=self.args
            )
            for s in range(brackets)
        ]

    def sort_solutions(self, solutions, fitnesses):

        if self.args.search == 'epsnet':
            population = [{'fitnesses': [x], 'index':i}
                          for i, x in enumerate(fitnesses)]
            fronts = fastNonDominatedSorting(
                population, self.n_objectives, self.constraints)
            population = epsnet_sort_all_fronts(
                fronts, self.n_objectives)  # best solutions first
            assert len(fitnesses) == len(population)
            sorted_solutions = [solutions[p['index']] for p in population]
            fitnesses = [fitnesses[p['index']] for p in population]
        
        elif self.args.search == 'HCBSOneRef':
            population = [{'fitnesses': [x], 'index':i}
                          for i, x in enumerate(fitnesses)]
            fronts = fastNonDominatedSorting(
                population, self.n_objectives, self.constraints)
            population = hcbs_sort_all_fronts_oneref(
                fronts, self.n_objectives)  # best solutions first
            assert len(fitnesses) == len(population)
            sorted_solutions = [solutions[p['index']] for p in population]
            fitnesses = [fitnesses[p['index']] for p in population]
        
        else:
            raise ValueError('objective type should be one of [\'epsnet\', \'HCBSMOTPE\'], found:%s' % self.args.search)


        return sorted_solutions, fitnesses

    def on_trial_result(
            self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        action = TrialScheduler.CONTINUE
        if self._time_attr not in result or self._metric not in result:
            return action
        if result[self._time_attr] >= self._max_t and self._stop_last_trials:
            action = TrialScheduler.STOP
        else:
            bracket = self._trial_info[trial.trial_id]
            action = bracket.on_result(
                trial, result[self._time_attr], result[self._metric]
            )
        if action == TrialScheduler.STOP:
            self._num_stopped += 1
        return action

    def on_trial_complete(
            self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ):
        if self._time_attr not in result or self._metric not in result:
            return
        bracket = self._trial_info[trial.trial_id]
        bracket.on_result(
            trial, result[self._time_attr], result[self._metric]
        )
        del self._trial_info[trial.trial_id]

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        if not self._metric or not self._metric_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.run()`".format(
                    self.__class__.__name__, self._metric, self._mode
                )
            )

        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._trial_info[trial.trial_id] = self._brackets[idx]

        self.trial_configs[trial.trial_id] = trial.config

        if self.TPESampler:
            max_milestone = -1
            min_points_in_model = 2*(len(trial.config))+1
            random_fraction = 1.0 / 6
            collected_solutions = None
            for bracket in self._brackets:
                for milestone, recorded in bracket._rungs:
                    solutions = recorded
                    if len(solutions) > min_points_in_model:
                        if milestone > max_milestone:
                            max_milestone = milestone
                            collected_solutions = solutions
                            print('bracket:{} | milestone:{} | #solutions:{}'.format(bracket, milestone, len(solutions)))

            if collected_solutions is None or np.random.uniform(0, 1) < random_fraction:
                pass
            else:
                solutions, fitness_values = [], []
                for trial_id in collected_solutions:
                    config = self.trial_configs[trial_id]
                    solutions.append(config)
                    fitness_values.append(collected_solutions[trial_id])

                for hp in self.search_space:
                    sampler = TPESampler(np.array([s[hp] for s in solutions]), fitness_values, choices=self.search_space[hp],
                                         random_seed=self.args.seed, n_objectives=self.n_objectives, constraints=self.constraints)
                    sampled_value = sampler.sample()
                    trial.config[hp] = sampled_value
                    print('sampled config:', trial.config)

    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner"
    ) -> Optional[Trial]:

        for trial in trial_runner.get_trials():
            if (
                    trial.status == Trial.PENDING
                    and trial_runner.trial_executor.has_resources_for_trial(trial)
            ):
                return trial

        for trial in trial_runner.get_trials():
            if (
                    trial.status == Trial.PAUSED
                    and trial_runner.trial_executor.has_resources_for_trial(trial)
            ):
                return trial
        return None


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

    for step in range(prev_step, args.steps):

        cur_epoch = step*EPOCHS_STEP

        train_seed = 1234 * args.seed + 5678 * cur_epoch + model_id
        weights_seed = 1234 * args.seed + model_id

        x = []
        for j in range(len(alphabet_array)):
            x.append(config['x%d' % j])
        
        print('train setup: config={} | model_id={} | cur_epoch={} | weights_seed={} | training_seed={}'.format(config, x, model_id,
              cur_epoch, weights_seed, train_seed))
        
        val_score, test_score, *_ = fitness_function.fitness(
            x, model_id, cur_epoch, weights_seed=weights_seed, train_seed=train_seed, cur_data_provider=data_provider)
        fitness = np.array(val_score)

        torch.cuda.empty_cache()
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


def runMOASHA(args, tpesampler=False):

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

    scheduler = MOASHA(
        time_attr="training_iteration",
        metric="mo_score",
        mode="max",
        max_t=int(args.steps),
        grace_period=1,
        reduction_factor=3,
        selection=args.search,
        base_folder=base_folder,
        args=args,
        TPESampler=tpesampler,
        search_space=search_space_with_lists
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(mytrain, args=args, MAX_EPOCHS=MAX_EPOCHS, alphabet_array=alphabet_array, base_folder=base_folder,
                                 data_provider=data_provider, function_create_model=function_create_model, hyperparameters=hyperparameters),
            resources={"cpu": 1, "gpu": 1.0/args.parallel_workers_per_gpu}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=-1,
            metric="mo_score",
            mode="max",
            time_budget_s=args.time_budget,
            # progress_reporter=CustomReporter()
        ),
        run_config=air.RunConfig(progress_reporter=CustomReporter()),
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
