from copy import copy
from typing import Callable


class LocalSearch:
    """
    Class that contains local search optimization.
    Arguments:
        fitness_func: external fitness func for estimation of
            solution effectivness
        fitness_coeff: coefficient that is used in fitness
            func, as power
        bin_capacity: maximum weight that bin can hold
    """
    def __init__(
        self,
        fitness_func: Callable,
        fitness_coeff: int,
        bin_capacity: int
    ):
        self.fitness_func = fitness_func
        self.fitness_coeff = fitness_coeff
        self.bin_capacity = bin_capacity

    def find_least_filled_bin(self, solution) -> int:
        """
        Scans solution and returns least filled bin's index
        """
        least_filled_bin_idx = None
        least_weight_diff = self.bin_capacity
        for bin_idx in range(len(solution)):
            local_sum = sum(solution[bin_idx])
            if self.bin_capacity - local_sum < least_weight_diff:
                least_weight_diff = self.bin_capacity - local_sum
                least_filled_bin_idx = bin_idx
        return least_filled_bin_idx

    def modify_solution(
        self,
        solution: list,
        free_n_bins: int
    ) -> tuple[list, list]:
        """
        "Opens" N least filled bins, freeing its items.
        Returns tuple of free items and solution without them
        """
        free_items = []
        modified_solution = copy(solution)
        for i in range(free_n_bins):
            idx = self.find_least_filled_bin(modified_solution)
            free_items += modified_solution[idx]
            del modified_solution[idx]
        return (free_items, modified_solution)

    def swap_two_by_two(self, _bin, pair, current_weight_diff):
        """
        Method of reallocating free items into filled bins.
        Tries to swap two filled items with two free items
        """
        modified_bin = copy(_bin)
        new_free = ()
        for i in range(1, len(_bin), 2):
            modified_bin = copy(_bin)
            new_free = (modified_bin[i - 1], modified_bin[i])
            modified_bin[i - 1] = pair[0]
            modified_bin[i] = pair[1]
            if (self.bin_capacity - sum(modified_bin)) < current_weight_diff:
                return new_free, modified_bin
        return new_free, modified_bin

    def swap_two_by_one(self, _bin, new_item, current_weight_diff):
        """
        Method of reallocating free items into filled bins.
        Tries to swap two filled items with one free item
        """
        modified_bin = copy(_bin)
        new_free = ()
        for i in range(1, len(_bin), 2):
            modified_bin = copy(_bin)
            new_free = (modified_bin[i - 1], modified_bin[i])
            modified_bin[i - 1] = 0
            modified_bin[i] = new_item
            if (self.bin_capacity - sum(modified_bin)) < current_weight_diff:
                modified_bin = [x for x in modified_bin if x > 0]
                return new_free, modified_bin
        return new_free, modified_bin

    def swap_one_by_one(self, _bin, new_item, current_weight_diff):
        """
        Method of reallocating free items into filled bins.
        Tries to swap one filled item with one free item
        """
        modified_bin = copy(_bin)
        new_free = 0
        for i in range(len(_bin)):
            modified_bin = copy(_bin)
            new_free = modified_bin[i]
            modified_bin[i] = new_item
            if (self.bin_capacity - sum(modified_bin)) < current_weight_diff:
                return new_free, modified_bin
        return new_free, modified_bin

    def first_fit_decreasing(self, solution, items):
        """
        Classic First Fit Decreasing algorithm for
        reallocating finally left free items into bins
        """
        items = sorted(items)
        for item in items:
            item_assigned = False
            for _bin in solution:
                free_space = self.bin_capacity - sum(_bin)
                if item <= free_space:
                    _bin.append(item)
                    item_assigned = True
                    break
            if not item_assigned:
                solution.append([item, ])
        return solution

    def _local_search(self, solution, free_n_bins):
        """
        Combined method that frees N bins of solution
        and tries to reallocate it, finally applying
        First Fit Decreasing algorithm
        """
        free_items, modified_solution = self.modify_solution(
            solution, free_n_bins
        )
        for bin_idx in range(len(modified_solution)):
            _bin = modified_solution[bin_idx]
            current_diff = self.bin_capacity - sum(_bin)
            # try two by two:
            new_free = ()
            for i in range(1, len(free_items), 2):
                pair = (free_items[i - 1], free_items[i])
                new_free, m_bin = self.swap_two_by_two(
                    _bin, pair, current_diff
                )
                if len(new_free) > 0:
                    free_items[i - 1] = new_free[i - 1]
                    free_items[i] = new_free[i]
                    modified_solution[bin_idx] = m_bin
                    break
            if len(new_free) > 0:
                continue

            # try two by one:
            for i in range(len(free_items)):
                new_item = free_items[i]
                new_free, m_bin = self.swap_two_by_one(
                    _bin, new_item, current_diff
                )
                if len(new_free) > 0:
                    free_items[i - 1] = new_free[i - 1]
                    free_items[i] = new_free[i]
                    modified_solution[bin_idx] = m_bin
                    break
            if len(new_free) > 0:
                continue

            # try one by one:
            new_free = 0
            for i in range(len(free_items)):
                new_item = free_items[i]
                new_free, m_bin = self.swap_one_by_one(
                    _bin, new_item, current_diff
                )
                if new_free > 0:
                    free_items[i] = new_free
                    modified_solution[bin_idx] = m_bin
                    break
            if new_free > 0:
                continue
        modified_solution = self.first_fit_decreasing(
            modified_solution, free_items
        )
        return modified_solution

    def find_better(self, solution: list[list], bins_to_open: int):
        """
        Public method for local search. Gets solution and tries to
        improve it via local search algorithm.
        Returns improved solution if it's possible, otherwise
        returns old, unmodified solution
        """
        original_fitness = self.fitness_func(solution)
        modified_solution = self._local_search(solution, bins_to_open)
        current_fitness = self.fitness_func(modified_solution)
        is_new_better = False
        while original_fitness < current_fitness:
            original_fitness = current_fitness
            modified_solution = self._local_search(solution, bins_to_open)
            current_fitness = self.fitness_func(modified_solution)
            is_new_better = True

        if is_new_better and len(solution) >= len(modified_solution):
            return modified_solution
        else:
            return solution
