# This code is adapted from https://github.com/automl/multi-obj-baselines/
import sys
import numpy as np
from search.mo_utils import *

eps = sys.float_info.epsilon


class UniformKernel:
    def __init__(self, n_choices):
        self.n_choices = n_choices

    def cdf(self, x):
        if 0 <= x <= self.n_choices - 1:
            return 1. / self.n_choices
        else:
            raise ValueError('The choice must be between {} and {}, but {} was given.'.format(
                0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        choice_one_hot = rng.multinomial(
            n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]


class AitchisonAitkenKernel:
    def __init__(self, choice, n_choices, top=0.9):
        self.n_choices = n_choices
        self.choice = choice
        self.top = top

    def cdf(self, x):
        if x == self.choice:
            return self.top
        elif 0 <= x <= self.n_choices - 1:
            return (1. - self.top) / (self.n_choices - 1)
        else:
            raise ValueError('The choice must be between {} and {}, but {} was given.'.format(
                0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        choice_one_hot = rng.multinomial(
            n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]


class GammaFunction:
    def __init__(self, gamma=0.10):
        self.gamma = gamma

    def __call__(self, x):
        # without upper bound for the number of lower samples
        return int(np.floor(self.gamma * x))


class NumericalParzenEstimator:
    def __init__(self, samples, lb, ub, weights_func, q=None, rule='james'):
        self.lb, self.ub, self.q, self.rule = lb, ub, q, rule
        self.weights, self.mus, self.sigmas = self._calculate(
            samples, weights_func)
        self.basis = [GaussKernel(m, s, lb, ub, q)
                      for m, s in zip(self.mus, self.sigmas)]

    def sample_from_density_estimator(self, rng, n_ei_candidates):
        samples = np.asarray([], dtype=float)
        while samples.size < n_ei_candidates:
            active = np.argmax(rng.multinomial(1, self.weights))
            drawn_hp = self.basis[active].sample_from_kernel(rng)
            samples = np.append(samples, drawn_hp)

        return samples if self.q is None else np.round(samples / self.q) * self.q

    def log_likelihood(self, xs):
        ps = np.zeros(xs.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.pdf(xs)

        return np.log(ps + eps)

    def basis_loglikelihood(self, xs):
        return_vals = np.zeros((len(self.basis), xs.size), dtype=float)
        for basis_idx, b in enumerate(self.basis):
            return_vals[basis_idx] += b.log_pdf(xs)

        return return_vals

    def _calculate(self, samples, weights_func):
        if self.rule == 'james':
            return self._calculate_by_james_rule(samples, weights_func)
        else:
            raise ValueError('unknown rule')

    def _calculate_by_james_rule(self, samples, weights_func):
        mus = np.append(samples, 0.5 * (self.lb + self.ub))
        sigma_bounds = [(self.ub - self.lb) /
                        min(100.0, mus.size), self.ub - self.lb]

        order = np.argsort(mus)
        sorted_mus = mus[order]
        original_order = np.arange(mus.size)[order]
        prior_pos = np.where(original_order == mus.size - 1)[0][0]

        sorted_mus_with_bounds = np.insert(
            [sorted_mus[0], sorted_mus[-1]], 1, sorted_mus)
        sigmas = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2],
                            sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
        sigmas = np.clip(sigmas, sigma_bounds[0], sigma_bounds[1])
        sigmas[prior_pos] = sigma_bounds[1]

        weights = weights_func(mus.size)
        weights /= weights.sum()

        return weights, mus, sigmas[original_order]


class CategoricalParzenEstimator:
    # note: this implementation has not been verified yet
    def __init__(self, samples, n_choices, weights_func, top=0.9):
        self.n_choices = n_choices
        self.mus = samples
        self.basis = [AitchisonAitkenKernel(
            c, n_choices, top=top) for c in samples]
        self.basis.append(UniformKernel(n_choices))
        self.weights = weights_func(samples.size + 1)
        self.weights /= self.weights.sum()

    def sample_from_density_estimator(self, rng, n_ei_candidates):
        basis_samples = rng.multinomial(
            n=1, pvals=self.weights, size=n_ei_candidates)
        basis_idxs = np.dot(basis_samples, np.arange(self.weights.size))
        return np.array([self.basis[idx].sample_from_kernel(rng) for idx in basis_idxs])

    def log_likelihood(self, values):
        ps = np.zeros(values.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.cdf_for_numpy(values)
        return np.log(ps + eps)

    def basis_loglikelihood(self, xs):
        return_vals = np.zeros((len(self.basis), xs.size), dtype=float)
        for basis_idx, b in enumerate(self.basis):
            return_vals[basis_idx] += b.log_cdf_for_numpy(xs)
        return return_vals


def default_weights(x):
    # default is uniform weights
    # we empirically confirmed that the recency weighting heuristic adopted in
    # Bergstra et al. (2013) seriously degrades performance in multiobjective optimization
    if x == 0:
        return np.asarray([])
    else:
        return np.ones(x)


class TPESampler:
    def __init__(self,
                 solutions,
                 fitness_values,
                 choices,
                 constraints,
                 n_objectives,
                 random_seed=42,
                 n_ei_candidates=24,
                 rule='james',
                 gamma_func=GammaFunction(),
                 weights_func=default_weights,
                 split_cache=None):
        self.solutions = np.array(solutions)
        self.fitness_values = np.array(fitness_values)
        self.choices = list(choices)

        self._random_state = np.random.RandomState(int(random_seed))
        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weights_func = weights_func
        self.opt = self.sample
        self.rule = rule
        if split_cache:
            self.split_cache = split_cache
        else:
            self.split_cache = dict()
        self.constraints = constraints
        self.n_objectives = n_objectives

    def _split_observations(self, solutions, fitness_values, n_lower):
        population = [{'fitnesses': [x], 'index':i}
                      for i, x in enumerate(fitness_values)]
        fronts = fastNonDominatedSorting(
            population, self.n_objectives, self.constraints)

        if n_lower == 0:
            return [], solutions

        good, bad = hcbs_oneref_asmotpe(
            fronts, n_lower, self.n_objectives, return_notselected=True)
        assert len(good)+len(bad) == len(population)

        print('good:{} \n bad:{}', good, bad)
        lower_indices = [x['index'] for x in good]
        upper_indices = [x['index'] for x in bad]
        
        return solutions[lower_indices], solutions[upper_indices]

    def _sample_categorical(self, lower_vals, upper_vals):
        choices = self.choices
        n_choices = len(choices)
        lower_vals = np.array([choices.index(val) for val in lower_vals])
        upper_vals = np.array([choices.index(val) for val in upper_vals])

        pe_lower = CategoricalParzenEstimator(
            lower_vals, n_choices, self.weights_func)
        pe_upper = CategoricalParzenEstimator(
            upper_vals, n_choices, self.weights_func)

        best_choice_idx = int(self._compare_candidates(pe_lower, pe_upper))
        return choices[best_choice_idx]

    def _compare_candidates(self, pe_lower, pe_upper):
        samples_lower = pe_lower.sample_from_density_estimator(
            self._random_state, self.n_ei_candidates)
        best_idx = np.argmax(
            pe_lower.log_likelihood(samples_lower) - pe_upper.log_likelihood(samples_lower))
        return samples_lower[best_idx]

    def sample(self):
        n_lower = self.gamma_func(len(self.solutions))
        print('n_lower:', n_lower)
        lower_vals, upper_vals = self._split_observations(
            self.solutions, self.fitness_values, n_lower)
        hp_value = self._sample_categorical(lower_vals, upper_vals)
        return hp_value
