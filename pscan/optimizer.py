import pygad
import numpy as np
import pandas as pd
from pscan.engine import backtest_dca
from typing import List, Any


class GeneticOptimizer:
    def __init__(self, prices_df: pd.DataFrame):
        self.prices_df = prices_df
        self.num_assets = len(prices_df.columns)

    def fitness_func(self, ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
        result = backtest_dca(self.prices_df, solution / np.sum(solution))

        roi = result['total_roi']
        mdd = abs(result['max_drawdown'])

        if roi <= 0:
            fitness = roi
        else:
            fitness = roi

        return float(fitness)

    def run(self) -> Any:
        ga_instance = pygad.GA(
            num_generations=500,
            num_parents_mating=7,
            fitness_func=self.fitness_func,
            sol_per_pop=self.num_assets,
            num_genes=self.num_assets,
            mutation_percent_genes=3,
            keep_elitism=2,
            on_generation=lambda ga: print(
                f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"),
            gene_type=float,

            init_range_low=0.0,
            init_range_high=1.0,
            gene_space={'low': -1.0, 'high': 1.0}
        )

        ga_instance.run()
        return ga_instance
