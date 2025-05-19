# WFO Performance Report: kama_crossover_otoco - WIFUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: kama_crossover_otoco
- Pair: WIFUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250515_151706

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': -114.4800*
```json
{
  "kama_period": 10,
  "kama_fast_ema": 2,
  "kama_slow_ema": 22,
  "indicateur_frequence_kama": "3min",
  "atr_period": 10,
  "atr_base_frequency": "1h",
  "sl_atr_mult": 1.0,
  "tp_atr_mult": 1.5,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | kama:6, kama:5, kama:31, kama:15min, atr:16, sl:2.00, tp:1.50, capi:0.80, marg:3.00 | 4 OOS trials. Best PnL: -102.69. PnLs: [-103, -290, -330...] |
| 1    | COMPLETED | kama:10, kama:2, kama:22, kama:3min, atr:10, sl:1.00, tp:1.50, capi:0.80, marg:3.00 | 8 OOS trials. Best PnL: -114.48. PnLs: [-114, -126, -143...] |

