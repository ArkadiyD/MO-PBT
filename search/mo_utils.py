import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances
import numbers
from copy import deepcopy

from pygmo import hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def calculateHypervolume(fitness_pairs, ref_point, n_objectives=2):
    ref_point = np.array(ref_point)
    A = np.array(fitness_pairs)
    hv = hypervolume(A)
    return hv.compute(ref_point)


def epsnet(fronts, to_select, n_objectives):
    selected_solutions = []
    not_selected_solutions = []
    added_solutions_indices = [[] for k in range(len(fronts))]
    for k, front in enumerate(fronts):
        front = list(front)
        front = sorted(front, key = lambda x:-x['fitnesses'][-1][0])
        print('front #%d' %k, ':', front)
        print('-'*100)
        if len(selected_solutions) + len(front) <= to_select:
            selected_solutions += front
            for index in range(len(front)):
                added_solutions_indices[k].append(index)

        else:
            cur_selected_indices = set([])

            while True:
                #print('selected', selected_solutions, len(selected_solutions))
                if len(selected_solutions) >= to_select or len(front) == len(cur_selected_indices):
                    break

                if len(selected_solutions) == 0:
                    selected_solutions.append(front[0])
                    cur_selected_indices.add(0)
                    added_solutions_indices[k].append(0)
                    continue

                points1 = np.array([x['fitnesses'][-1]
                                   for x in front]).reshape(-1, n_objectives)
                points2 = np.array(
                    [x['fitnesses'][-1] for x in selected_solutions]).reshape(-1, n_objectives)
                distances = euclidean_distances(points1, points2)
                cur_min_distances = np.min(distances, axis=1)

                ind_with_max_dist = -1
                max_dist = -float("inf")
                for j in range(len(front)):
                    if j not in cur_selected_indices:
                        if cur_min_distances[j] > max_dist:
                            max_dist = cur_min_distances[j]
                            ind_with_max_dist = j

                selected_solutions.append(front[ind_with_max_dist])
                cur_selected_indices.add(ind_with_max_dist)
                added_solutions_indices[k].append(ind_with_max_dist)

        if len(selected_solutions) >= to_select:
            break

    for k, front in enumerate(fronts):
        for i in range(len(front)):
            if i not in added_solutions_indices[k]:
                not_selected_solutions.append(front[i])

    print('selected_solutions', selected_solutions)
    return selected_solutions, not_selected_solutions


def epsnet_sort_all_fronts(fronts, n_objectives):
    sorted_solutions = []
    for k, front in enumerate(fronts):
        front = list(front)
        front = sorted(front, key = lambda x:-x['fitnesses'][-1][0])
        print('front #%d' %k, ':', front)
        print('-'*100)

        cur_selected_indices = set([])

        points1 = np.array([x['fitnesses'][-1]
                           for x in front]).reshape(-1, n_objectives)

        while len(cur_selected_indices) < len(front):

            if len(sorted_solutions) == 0:
                sorted_solutions.append(front[0])
                cur_selected_indices.add(0)
                continue

            points2 = np.array([x['fitnesses'][-1]
                               for x in sorted_solutions]).reshape(-1, n_objectives)
            distances = euclidean_distances(points1, points2)
            cur_min_distances = np.min(distances, axis=1)

            ind_with_max_dist = -1
            max_dist = -float("inf")
            for j in range(len(front)):
                if j not in cur_selected_indices:
                    if cur_min_distances[j] > max_dist:
                        max_dist = cur_min_distances[j]
                        ind_with_max_dist = j

            sorted_solutions.append(front[ind_with_max_dist])
            cur_selected_indices.add(ind_with_max_dist)

    return sorted_solutions


def hcbs_oneref_asmotpe(fronts, to_select, n_objectives, return_notselected=True):
    selected_solutions = []
    not_selected_solutions = []
    added_solutions_indices = [[] for k in range(len(fronts))]

    for k, front in enumerate(fronts):
        front = list(front)
        print('front #%d' %k, ':', front)
        print('-'*100)
        if len(selected_solutions) + len(front) <= to_select:
            selected_solutions += front
            for index in range(len(front)):
                added_solutions_indices[k].append(index)

        else:
            all_scores = [s['fitnesses'][-1] for s in front]
            all_scores = -np.array(all_scores).reshape(-1, n_objectives)
            worst_point = np.max(all_scores, axis=0)
            ref_point = 0.9 * worst_point  # case: value < 0
            ref_point[ref_point == 0.0] = 0.1

            cur_selected_indices = set([])
            scores = -np.array([s['fitnesses'][-1]
                               for s in front]).reshape(-1, n_objectives)
            print('ref_point', ref_point)
            scores = list(scores)
            contributions = []
            for j in range(len(scores)):
                contributions.append(calculateHypervolume(
                    [scores[j]], ref_point, n_objectives))
            S = []

            while True:
                #print('selected', selected_solutions, len(selected_solutions))
                if len(selected_solutions) >= to_select or len(front) == len(cur_selected_indices):
                    break

                hv_S = 0
                if len(S) > 0:
                    hv_S = calculateHypervolume(S, ref_point, n_objectives)
                index = np.argmax(contributions)
                contributions[index] = -1e9  # mark as already selected
                for j in range(len(contributions)):
                    if contributions[j] == -1e9:
                        continue
                    p_q = np.max([scores[index], scores[j]], axis=0)
                    contributions[j] = contributions[j] - \
                        (calculateHypervolume(
                            S + [p_q], ref_point, n_objectives) - hv_S)
                S = S + [scores[index]]

                selected_solutions.append(front[index])
                added_solutions_indices[k].append(index)
                cur_selected_indices.add(index)

        if len(selected_solutions) >= to_select:
            break
    print('added solutions:', added_solutions_indices)
    for k, front in enumerate(fronts):
        for i in range(len(front)):
            if i not in added_solutions_indices[k]:
                not_selected_solutions.append(front[i])

    if not return_notselected:
        return selected_solutions
    else:
        return selected_solutions, not_selected_solutions


def fastNonDominatedSorting(population, n_objectives, constraints):
    sorting = NonDominatedSorting()

    solutions = np.array([x['fitnesses'][-1]
                         for x in population]).reshape(-1, n_objectives)

    if constraints is not None:
        for j in range(len(constraints)):
            C = constraints[j]
            if isinstance(C, numbers.Number):
                mask = solutions[:, j] <= C
                if len(mask):
                    violation = np.array(C-solutions[mask, j]).reshape(-1, 1)
                    solutions[mask] = np.minimum(solutions[mask], 0.0)
                    solutions[mask] -= violation

    solutions = -solutions
    fronts_ = sorting.do(solutions, return_rank=False)
    fronts = []
    for front_ in fronts_:
        fronts.append([population[i] for i in front_])
    return fronts


def scalarize(type, f, alpha, n_objectives):
    f = np.array(f).reshape(-1, n_objectives)

    if isinstance(alpha, float) or isinstance(alpha, int):
        weights = [alpha, 1-alpha]
    weights = np.array(alpha)

    if type == 'WS':
        scalar = np.sum(f * weights, axis=1)

    elif type == 'TE':
        scalar = np.min(f * weights, axis=1)

    elif type == 'Parego':
        TE_part = scalarize('TE', f, alpha, n_objectives)
        WS_part = scalarize('WS', f, alpha, n_objectives)
        scalar = TE_part + 0.05*WS_part

    elif type == 'Golovin':
        scalars = (f / weights)**2
        scalar = np.min(scalars, axis=1)

    else:
        raise ValueError('scalarization type should be one of [\'WS\', \'TE\', \'Parego\', \'Golovin\'], found:%s' % type)

    return scalar


def mutation(type, solution, alphabet, population=None):
    if type == 'random':
        return random_mutation(solution, alphabet)
    elif type == 'PBA':
        return PBA_mutation(solution, alphabet)
    else:
        raise ValueError('mutation type should be one of [\'random\', \'PBA\'], found:%s' % type)


def PBA_mutation(solution, alphabet_array):
    mutated_solution = deepcopy(solution)
    for k in range(len(solution['hyperparameters'])):
        if np.random.uniform(0, 1) < 0.2:
            mutated_solution['hyperparameters'][k] = np.random.randint(
                alphabet_array[k])
        else:
            diff = np.random.choice([0, 1, 2, 3])
            if np.random.uniform(0, 1) < 0.5:
                mutated_solution['hyperparameters'][k] += diff
            else:
                mutated_solution['hyperparameters'][k] -= diff
        mutated_solution['hyperparameters'][k] = np.clip(mutated_solution['hyperparameters'][k], 0,
                                                         alphabet_array[k] - 1)

    return mutated_solution


def random_mutation(solution, alphabet, prob='auto'):
    '''
    Simple bit-wise mutation (bits flipped with probability=prob)
    '''
    mutated_solution = deepcopy(solution)

    if prob == 'auto':
        prob = 1.0 / len(solution['hyperparameters'])

    for i in range(len(solution['hyperparameters'])):
        if np.random.uniform(0, 1) < prob:
            mutated_solution['hyperparameters'][i] = np.random.randint(
                alphabet[i])

    return mutated_solution


def assign_actual_fitness(fitness_values, objective, n_objectives, constraints, fitness_values_archive=None, results_folder=None):

    if objective == 'RandomSearch':
        return fitness_values

    elif objective == 'objective1':
        fitness_values = [x[0] for x in fitness_values]

    elif objective == 'objective2':
        fitness_values = [x[1] for x in fitness_values]

    elif objective == 'objective3':
        fitness_values = [x[2] for x in fitness_values]

    elif 'max_scalarization' in objective:
        alphas = np.random.uniform(
            0, 1, (len(fitness_values), 100, n_objectives))
        alphas /= np.sum(alphas,
                         axis=2).reshape(alphas.shape[0], alphas.shape[1], -1)

        type_scalarization = objective.replace('max_scalarization_', '')

        for i in range(len(fitness_values)):

            scalars = scalarize(type_scalarization,
                                fitness_values[i], alphas[i], n_objectives)
            fitness_values[i] = np.max(scalars)

    elif 'random_scalarization' in objective:
        alphas = np.random.uniform(0, 1, (len(fitness_values), n_objectives))
        alphas /= np.sum(alphas, axis=1).reshape(alphas.shape[0], -1)
        type_scalarization = objective.replace('random_scalarization_', '')

        fitness_values = scalarize(
            type_scalarization, fitness_values, alphas, n_objectives)

    elif objective == 'epsnet':
        deepcopy(fitness_values)
        population = [{'fitnesses': [x], 'index':i}
                      for i, x in enumerate(fitness_values)]
        fronts = fastNonDominatedSorting(
            population, n_objectives, constraints)
        population = epsnet_sort_all_fronts(fronts, n_objectives)
        assert len(fitness_values) == len(population)
        for i in range(len(population)):
            fitness_values[population[i]['index']] = len(population)-i
        
    else:
        raise ValueError('search type should be one of [\'objective[k]\', \'epsnet\', \'random_scalarization\', \'max_scalarization\'], found:%s' % objective)

    return fitness_values
