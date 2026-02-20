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
        """
        Фитнес-функция: максимизирует ROI и минимизирует просадку.
        """
        # Нормализация весов (сумма = 1.0)
        weights = solution / np.sum(solution) if np.sum(solution) > 0 else np.zeros_like(solution)
        
        # Запуск бэктеста
        result = backtest_dca(self.prices_df, weights)
        
        # Целевая функция: ROI + штраф за просадку
        # max_drawdown отрицательный, поэтому используем его со знаком + или через 1 + drawdown
        # Мы хотим минимизировать просадку (сделать ее ближе к 0)
        # Если просадка -0.5, то штраф может быть значительным.
        
        roi = result['total_roi']
        mdd = abs(result['max_drawdown'])
        
        # Целевая функция: максимизируем ROI, сильно штрафуем за просадку
        # Используем экспоненциальный штраф за просадку или простое умножение
        # Мы хотим избежать портфелей с MDD > 50%
        
        # Если просадка > 95% (практически полная потеря), фитнес должен быть очень низким
        if mdd > 0.95:
            return -100.0

        # Базовый фитнес - ROI. 
        # Если ROI положительный, умножаем на (1 - mdd)^2 для усиления штрафа за риск
        if roi > 0:
            fitness = roi * (1 - mdd)**2
        else:
            # Если ROI отрицательный, фитнес равен ROI (отрицательное число)
            fitness = roi * (1 + mdd)
            
        return float(fitness)

    def run(self, num_generations: int = 50, sol_per_pop: int = 20) -> Any:
        """
        Запускает ГА.
        """
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=10,
            fitness_func=self.fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=self.num_assets,
            init_range_low=0.0,
            init_range_high=1.0,
            mutation_percent_genes=10,
            on_generation=lambda ga: print(f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]:.4f}"),
            gene_type=float,
            # Ограничение значений генов от 0 до 1
            gene_space={'low': 0.0, 'high': 1.0}
        )
        
        ga_instance.run()
        return ga_instance
