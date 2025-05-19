# WFO Performance Report: ma_crossover_strategy - AIXBTUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: ma_crossover_strategy
- Pair: AIXBTUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250514_205812

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': -2.5200*
```json
{
  "fast_ma_period": 6,
  "slow_ma_period": 47,
  "ma_type": "hma",
  "indicateur_frequence_ma_rapide": "3min",
  "indicateur_frequence_ma_lente": "30min",
  "atr_period": 19,
  "atr_base_frequency": "1h",
  "sl_atr_multiplier": 3.0,
  "tp_atr_multiplier": 2.0,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | fast:14, slow:41, rapide:3min, lente:30min, atr:15, sl:2.00, tp:2.50, capi:0.80, marg:3.00 | 6 OOS trials. Best PnL: 1079.67. PnLs: [1080, 364, -59...] |
| 1    | COMPLETED | fast:6, slow:47, rapide:3min, lente:30min, atr:19, sl:3.00, tp:2.00, capi:0.80, marg:3.00 | 3 OOS trials. Best PnL: -2.52. PnLs: [-3, -211, -272...] |

