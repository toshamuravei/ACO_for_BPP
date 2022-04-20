from aco import ACO_for_BPP


if __name__ == "__main__":
    FILENAME = "120_items.txt"

    with open(FILENAME) as file:
        items = [int(line.rstrip()) for line in file]

    run_params = {
        "items": items,
        "n_ants": 120,
        "n_iterations": 100,
        "bin_max_weight": 150,
        "evaporation_coeff": 0.95,
        "beta_coeff": 5,
        "fitness_func_coeff": 2,
        "bins_to_open": 2,
        "random_state_seed": None,
    }

    colony = ACO_for_BPP(**run_params)
    best_solution = colony.solve_BPP()
    print(
        "shotest_path_120: {}".format(best_solution),
        ' ',
        len(best_solution[0])
    )
