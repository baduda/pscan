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
        clean_solution = np.where(solution < 0, 0, solution)
        clean_solution = np.where(clean_solution < 0.01, 0, clean_solution)

        total_weight = np.sum(clean_solution)
        if total_weight == 0:
            return -9999.0  # Жесткий штраф за портфель без позиций

        normalized_weights = clean_solution / total_weight

        result = backtest_dca(self.prices_df, normalized_weights)

        roi = result['total_roi']
        mdd = abs(result['max_drawdown'])
        sharpe = result['sharpe_ratio']
        calmar = roi / (mdd + 0.01)

        fitness = calmar + (sharpe * 0.5)

        return float(fitness)

    def run(self) -> Any:
        ga_instance = pygad.GA(
            num_generations=200,
            num_parents_mating=7,
            fitness_func=self.fitness_func,
            sol_per_pop=self.num_assets,
            num_genes=self.num_assets,
            mutation_percent_genes=4,
            keep_elitism=4,
            on_generation=lambda ga: print(
                f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"),
            gene_type=float,

            init_range_low=-1.0,
            init_range_high=1.0,
            gene_space={'low': -1.0, 'high': 1.0}
        )

        ga_instance.run()
        return ga_instance
