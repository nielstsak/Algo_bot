# WFO Performance Report: kama_crossover_otoco - AIXBTUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: kama_crossover_otoco
- Pair: AIXBTUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250514_205812

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 158.5800*
```json
{
  "kama_period": 13,
  "kama_fast_ema": 3,
  "kama_slow_ema": 20,
  "indicateur_frequence_kama": "15min",
  "atr_period": 16,
  "atr_base_frequency": "5min",
  "sl_atr_mult": 0.5,
  "tp_atr_mult": 3.0,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | kama:16, kama:4, kama:36, kama:3min, atr:16, sl:3.00, tp:1.50, capi:0.80, marg:3.00 | 2 OOS trials. Best PnL: 886.85. PnLs: [887, -49...] |
| 1    | COMPLETED | kama:13, kama:3, kama:20, kama:15min, atr:16, sl:0.50, tp:3.00, capi:0.80, marg:3.00 | 5 OOS trials. Best PnL: 158.58. PnLs: [159, 28, -81...] |

