# WFO Performance Report: adx_direction_otoco - AIXBTUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: adx_direction_otoco
- Pair: AIXBTUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250514_205812

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 234.0400*
```json
{
  "adx_period": 20,
  "adx_threshold": 21.0,
  "indicateur_frequence_adx": "3min",
  "atr_period": 15,
  "atr_base_frequency": "30min",
  "sl_atr_mult": 2.0,
  "tp_atr_mult": 3.0,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | adx:23, adx:15.50, adx:3min, atr:18, sl:3.00, tp:3.00, capi:0.80, marg:3.00 | 5 OOS trials. Best PnL: -23.11. PnLs: [-23, -149, -207...] |
| 1    | COMPLETED | adx:20, adx:21.00, adx:3min, atr:15, sl:2.00, tp:3.00, capi:0.80, marg:3.00 | 7 OOS trials. Best PnL: 234.04. PnLs: [234, 163, -168...] |

