# WFO Performance Report: psar_reversal_otoco - AIXBTUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: psar_reversal_otoco
- Pair: AIXBTUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250514_205812

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 400.4400*
```json
{
  "psar_step": 0.03,
  "psar_max_step": 0.26,
  "indicateur_frequence_psar": "5min",
  "atr_period": 18,
  "atr_base_frequency": "1h",
  "sl_atr_mult": 3.0,
  "tp_atr_mult": 2.5,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | psar:0.01, psar:0.14, psar:3min, atr:12, sl:3.00, tp:3.00, capi:0.80, marg:3.00 | 3 OOS trials. Best PnL: 166.43. PnLs: [166, -172, -172...] |
| 1    | COMPLETED | psar:0.03, psar:0.26, psar:5min, atr:18, sl:3.00, tp:2.50, capi:0.80, marg:3.00 | 10 OOS trials. Best PnL: 400.44. PnLs: [400, 400, 283...] |

