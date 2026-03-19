import pygad
import cma
import numpy as np
import pandas as pd
from pscan.engine import backtest_dca
from typing import List, Any


def _normalize_weights(weights: np.ndarray, min_weight: float = 0.01, max_weight: float = 0.7) -> np.ndarray:
    w = np.where(weights < 0, 0, weights).astype(float)
    total = np.sum(w)
    if total == 0:
        return None
    w = w / total
    # Cut small positions first
    w = np.where(w < min_weight, 0, w)
    total2 = np.sum(w)
    if total2 == 0:
        return None
    w = w / total2
    # Cap max weight with redistribution (simple clip + renorm undoes the cap)
    for _ in range(20):
        over = w > max_weight + 1e-9
        if not over.any():
            break
        extra = np.sum(w[over] - max_weight)
        w[over] = max_weight
        under = (~over) & (w > 0)
        if not under.any():
            break
        w[under] += extra / under.sum()
    total3 = np.sum(w)
    if total3 == 0:
        return None
    return w / total3


def _concentration_penalty(weights: np.ndarray) -> float:
    """Smooth penalty when fewer than 3 non-zero positions."""
    n = int(np.sum(weights > 0))
    return -max(0, 3 - n) ** 2


def _calc_fitness(result: dict) -> float:
    roi = result['total_roi']
    mdd = abs(result['max_drawdown'])
    sortino = result['sortino_ratio']

    if not np.isfinite(roi): roi = -1.0
    if not np.isfinite(mdd) or mdd == 0: mdd = 1.0
    if not np.isfinite(sortino): sortino = 0.0

    calmar = roi / (mdd + 0.01)
    return (calmar * 2) + sortino


class GeneticOptimizer:
    def __init__(self, prices_df: pd.DataFrame):
        self.prices_df = prices_df
        self.num_assets = len(prices_df.columns)

    def fitness_func(self, ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
        normalized_weights = _normalize_weights(solution)
        if normalized_weights is None:
            return -9999.0

        result = backtest_dca(self.prices_df, normalized_weights)
        fitness = _calc_fitness(result) + _concentration_penalty(normalized_weights)
        if not np.isfinite(fitness):
            return -1e6
        return float(fitness)

    def run(self) -> Any:
        ga_instance = pygad.GA(
            num_generations=500,
            num_parents_mating=6,
            fitness_func=self.fitness_func,
            sol_per_pop=self.num_assets,
            num_genes=self.num_assets,
            mutation_percent_genes=4,
            keep_elitism=2,
            on_generation=lambda ga: print(
                f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"),
            gene_type=float,

            init_range_low=0.0,
            init_range_high=1.0,
            gene_space={'low': 0.0, 'high': 1.0}
        )

        ga_instance.run()
        return ga_instance


class _CMAResult:
    def __init__(self, weights: np.ndarray, fitness: float):
        self._weights = weights
        self._fitness = fitness

    def best_solution(self):
        return (self._weights, self._fitness, 0)


class CMAOptimizer:
    def __init__(self, prices_df: pd.DataFrame):
        self.prices_df = prices_df
        self.num_assets = len(prices_df.columns)

    def _objective(self, weights: np.ndarray) -> float:
        normalized = _normalize_weights(weights)
        if normalized is None:
            return 9999.0
        result = backtest_dca(self.prices_df, normalized)
        return -(_calc_fitness(result) + _concentration_penalty(normalized))

    def run(self) -> _CMAResult:
        x0 = np.ones(self.num_assets) / self.num_assets
        sigma0 = 0.1
        opts = cma.CMAOptions()
        opts['maxiter'] = 1000
        opts['tolx'] = 1e-4
        opts['verbose'] = -9
        opts['bounds'] = [0, 1]
        opts['tolfun'] = 1e-4    # остановится если fitness почти не меняется
        opts['CMA_diagonal'] = True

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = [self._objective(x) for x in solutions]
            es.tell(solutions, fitnesses)
            if es.result.iterations % 50 == 0:
                print(f"Iteration {es.result.iterations}: Best Fitness = {-es.result.fbest:.4f}")

        return _CMAResult(es.result.xbest, -es.result.fbest)
