# WFO Performance Report: psar_reversal_otoco - PEPEUSDC ( optimisation new_strat)

## WFO Configuration
- Strategy: psar_reversal_otoco
- Pair: PEPEUSDC
- Context:  optimisation new_strat
- WFO Run Timestamp: 20250519_060729

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 1706.1100*
```json
{
  "psar_step": 0.02,
  "psar_max_step": 0.3,
  "indicateur_frequence_psar": "1h",
  "atr_period": 19,
  "atr_base_frequency": "15min",
  "sl_atr_mult": 1.75,
  "tp_atr_mult": 2.75,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | psar:0.04, psar:0.30, psar:1h, atr:20, sl:2.00, tp:3.25, capi:0.80, marg:3.00 | 4 OOS trials. Best PnL: 1195.73. PnLs: [1196, 796, -10000...] |
| 1    | COMPLETED | psar:0.02, psar:0.30, psar:1h, atr:19, sl:1.75, tp:2.75, capi:0.80, marg:3.00 | 10 OOS trials. Best PnL: 1706.11. PnLs: [1706, 1649, 1585...] |

