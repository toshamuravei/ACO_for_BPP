from typing import Optional

import numpy as np

from local_search import LocalSearch


class ACO_for_BPP:
    """
    Class that holds logic of Ant Colony Optimization for solving
    Bin Packing Problem.
    Arguments:
        items - list of weighted items (ints) to be placed in bins
        n_ants - number of artificial ants to be used in colony simulation
        n_iterations - number of ant "runs", algorithm iterations
        bin_max_weight - integer capacity of single bin
        evaporations_coeff - so-called "rho",it is pheromone evaporation speed
        beta_coeff - coeff balancing "influence" of pheromones & fitness func
        fitness_func_coeff - coeff (power) for fitness func
        random_state_seed - seed for random state of decision making
    """
    PHEROMONES_COEFF = 20
    INITIAL_GLOBAL_BEST = ("", 0)
    ITERATION_LOG_MSG = "Iteration #{i_num}; Shortest path length: {path_len}"
    LONG_SCALAR_COEFF = 0.00000001
    BETA_LIMIT_FOR_LONG_SCALARS = 10

    def __init__(
        self,
        items: list[int],
        n_ants: int,
        n_iterations: int,
        bin_max_weight: int,
        evaporation_coeff: float,
        beta_coeff: int,
        fitness_func_coeff: int,
        bins_to_open: int,
        random_state_seed: Optional[list]
    ):
        self.items = items
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.bin_max_weight = bin_max_weight
        self.evaporation_coeff = evaporation_coeff
        self.beta_coeff = beta_coeff
        self.fitness_func_coeff = fitness_func_coeff
        self.long_scalar_coeff = self._init_long_scalar_coeff(beta_coeff)
        self.state = self._init_state(random_state_seed)
        self.pheromones_matrix: np.ndarray = self._init_pheromones(items)
        self.bins_to_open_localy = bins_to_open
        self.local_search = LocalSearch(
            fitness_func=self._fitness_solution,
            fitness_coeff=self.fitness_func_coeff,
            bin_capacity=self.bin_max_weight
        )

    def _init_long_scalar_coeff(self, beta) -> float:
        """
        Coefficient that is needed in decision making
        calculation. If beta coeff is high enough (more than 5)
        calculations may fall into overflowing long_scalar's
        value and freeze. Ignoring such warning could lead into
        mistakes in calculations. So in that cases reducing
        coefficient applied.
        """
        if self.beta_coeff < self.BETA_LIMIT_FOR_LONG_SCALARS:
            return 1.0
        else:
            return self.LONG_SCALAR_COEFF

    def _init_state(self, seed: Optional[list]):
        """
        Returns random state to be uniformed in
        decision making
        """
        return np.random.RandomState(seed)

    def _init_pheromones(self, items: list[int]) -> np.ndarray:
        """
        Returns phermones matrix, consisting of 1.0, because
        initially all paths should be equally attractive
        """
        return np.ones(
            [max(items) + 1, max(items) + 1],
            dtype=np.float128
        ) * self.PHEROMONES_COEFF

    def solve_BPP(self) -> tuple[list, float]:
        """
        Main public method that holds all algo's logic.
        1. Ants running across items matrix, creating
        iterations best solution
        2. Pheromones evaporates
        3. Only the best iteration solution leaves
        new pheromones
        4. If current iteration best is better than
        global best, it becomes new global best
        """
        iteration_best = None
        global_best = self.INITIAL_GLOBAL_BEST

        for iteration in range(self.n_iterations):
            all_paths = self._run_ants()
            self._evaporate_pheromones()
            iteration_best = max(all_paths, key=lambda x: x[1])
            if iteration % 5 == 0 and iteration > 0:
                self._update_pheromones(global_best)
            else:
                self._update_pheromones(iteration_best)
            self._log_iteration(iteration, iteration_best)
            if iteration_best[1] > global_best[1]:
                global_best = iteration_best

        return global_best

    def _run_ants(self) -> list[tuple]:
        """
        Returns solutions that are built from
        ants run. Fitness function applied
        """
        solutions = [None] * self.n_ants
        for i in range(self.n_ants):
            path = self._build_solution()
            path = self.local_search.find_better(
                path, self.bins_to_open_localy
            )
            solutions[i] = (path, self._fitness_solution(path))
        return solutions

    def _evaporate_pheromones(self) -> None:
        """
        Multiplies all pheromones by evaporation
        coefficient (rho)
        """
        self.pheromones_matrix * self.evaporation_coeff

    def _update_pheromones(self, new_best):
        """
        Gets best solution and updates pheromones with
        that ant's trail
        """
        best_ant = new_best[0]
        best_fitness = new_best[1]

        for _bin in best_ant:
            for i in range(len(_bin)):
                for j in range(i+1, len(_bin)):
                    self.pheromones_matrix[_bin[i]][_bin[j]] += best_fitness
                    self.pheromones_matrix[_bin[j]][_bin[i]] += best_fitness

    def _log_iteration(self, iter_num, iter_best):
        """
        Simple "print"-based iteration log function
        """
        print(self.ITERATION_LOG_MSG.format(
            i_num=iter_num,
            path_len=len(iter_best[0])
            )
        )

    def _build_solution(self):
        """
        Returns a path that is made by ant.
        Path is a path on items-bins matrix,
        that represents ant's try/solution of
        items packing between bins
        """
        items = list(np.copy(self.items))
        path = []
        for i in range(len(self.items)):
            path_row = self._pack_bin(items)
            path.append(path_row)
            if len(items) == 0:
                break
        return path

    def _fitness_solution(self, solution) -> float:
        """
        Simple fitness functions: SUM(Fi/C) / N
        Where is:
        Fi - total weight of items in i-th bin
        C  - bin weight capacity
        N  - number of bins used
        """
        bins_sum = 0
        for _bin in solution:
            items = 0
            for item in _bin:
                items += item
            bins_sum += (items/self.bin_max_weight) ** self.fitness_func_coeff
        return bins_sum/len(solution)

    def _pack_bin(self, items: list):
        """
        Ant's bin packing method. Ant tries to place items into bins
        deciding it with existing pheromones matrix.
        Returns packed items in bin
        """
        bin_items = []
        _bin = 0
        while len(items) > 0 and (self.bin_max_weight - _bin) > min(items):
            for i in items:
                bin_space_left = self.bin_max_weight - _bin
                if self._packing_allowed(bin_space_left, i, bin_items, items):
                    bin_items.append(i)
                    _bin += i
                    items.remove(i)
                    break
                if len(items) == 0 or self.bin_max_weight - _bin < min(items):
                    break
        return bin_items

    def _packing_allowed(self, free_space, item, bin_items, items) -> bool:
        """
        Returns answer (True/False) on question:
        "Is THIS item should be placed in THIS bin?"
        Answer is based on pheromones decision calculation and bin's
        fullness
        """
        if item > free_space:
            return False

        denominator = 0
        for i in items:
            if i <= free_space:
                denominator += self._decision_formula_term(i, bin_items)
        numerator = self._decision_formula_term(item, bin_items)
        state_uniform = self.state.uniform() * self.long_scalar_coeff
        return state_uniform < (numerator/denominator)

    def _decision_formula_term(self, item, bin_items) -> float:
        """
        Method for decision formula calculation.
        Both terms - numerator and denominator uses same formula but
        different items
        """
        d = self._tao_coeff(item, bin_items) * (item ** self.beta_coeff)
        return d * self.long_scalar_coeff

    def _tao_coeff(self, item, bin_items) -> float:
        """
        Tao coefficient calculation, that is used in decision formula
        """
        pheromones_sum = 0
        if len(bin_items) == 0:
            return 1

        for i in bin_items:
            pheromones_sum += self.pheromones_matrix[i][item]

        return pheromones_sum/len(bin_items)
