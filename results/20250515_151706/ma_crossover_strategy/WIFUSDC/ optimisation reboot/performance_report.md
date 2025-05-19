# WFO Performance Report: ma_crossover_strategy - WIFUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: ma_crossover_strategy
- Pair: WIFUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250515_151706

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 320.7700*
```json
{
  "fast_ma_period": 10,
  "slow_ma_period": 81,
  "ma_type": "wma",
  "indicateur_frequence_ma_rapide": "3min",
  "indicateur_frequence_ma_lente": "15min",
  "atr_period": 13,
  "atr_base_frequency": "1h",
  "sl_atr_multiplier": 0.5,
  "tp_atr_multiplier": 2.5,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | fast:12, slow:81, rapide:15min, lente:30min, atr:16, sl:2.00, tp:2.50, capi:0.80, marg:3.00 | 5 OOS trials. Best PnL: -124.93. PnLs: [-125, -165, -234...] |
| 1    | COMPLETED | fast:10, slow:81, rapide:3min, lente:15min, atr:13, sl:0.50, tp:2.50, capi:0.80, marg:3.00 | 5 OOS trials. Best PnL: 320.77. PnLs: [321, -53, -125...] |

