## Commands
```bash
python run.py              # Run GA optimization + DCA backtest + plot
python -m pscan.dca_ranker # Rank coins by DCA deviation metrics
python -m pscan.dca_report # Generate portfolio report
pip install -r requirements.txt  # Install dependencies
```

## Architecture
```
pscan/
  main.py        — entry point: load data → GA optimize → backtest → plot
  data_loader.py — fetch OHLCV via ccxt (Binance), cache to data/crypto_data.csv
  optimizer.py   — GeneticOptimizer (pygad): maximizes Sharpe ratio
  engine.py      — backtest_dca(): weekly DCA backtest, computes ROI/MDD/Sharpe
  dca_ranker.py  — standalone script: rank coins by DCA deviation
  dca_report.py  — generate report file

data/            — CSV price cache (crypto_data.csv)
logs/            — optimization and ranking logs
plots/           — output charts (optimization_result.png)
```

## Key Config (pscan/main.py)
- `target_symbols` — list of 27 USDT pairs to optimize
- `excluded_symbols` — exclusions (currently empty)
- `start_date` — '2021-01-01' (backtest start)
- `weekly_investment` — 100 USDT/week across the whole portfolio (engine.py)

## Gotchas
- CSV cache is invalidated if 'USDT/USDT' column is missing — data re-downloaded
- GA requires at least 2 assets; weights < 0.01 are zeroed before normalization
- matplotlib backend forced to Agg (non-interactive) in run.py
- GA: 300 generations, 10×N_assets population — slow with large symbol lists
- Fitness function uses Sharpe ratio only (calmar is computed but unused)