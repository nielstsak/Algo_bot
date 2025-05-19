# WFO Performance Report: ma_crossover_strategy - PEPEUSDC ( optimisation new_strat)

## WFO Configuration
- Strategy: ma_crossover_strategy
- Pair: PEPEUSDC
- Context:  optimisation new_strat
- WFO Run Timestamp: 20250519_060729

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 107.1300*
```json
{
  "fast_ma_period": 14,
  "slow_ma_period": 75,
  "ma_type": "hma",
  "indicateur_frequence_ma_rapide": "5min",
  "indicateur_frequence_ma_lente": "30min",
  "atr_period": 19,
  "atr_base_frequency": "30min",
  "sl_atr_multiplier": 2.5,
  "tp_atr_multiplier": 2.5,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | fast:14, slow:20, rapide:3min, lente:5min, atr:13, sl:3.00, tp:1.75, capi:0.80, marg:3.00 | 2 OOS trials. Best PnL: -10000.00. PnLs: [-10000, -10000...] |
| 1    | COMPLETED | fast:14, slow:75, rapide:5min, lente:30min, atr:19, sl:2.50, tp:2.50, capi:0.80, marg:3.00 | 1 OOS trials. Best PnL: 107.13. PnLs: [107...] |

