# WFO Performance Report: adx_direction_otoco - WIFUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: adx_direction_otoco
- Pair: WIFUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250515_151706

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': -273.3800*
```json
{
  "adx_period": 22,
  "adx_threshold": 15.0,
  "indicateur_frequence_adx": "30min",
  "atr_period": 20,
  "atr_base_frequency": "30min",
  "sl_atr_mult": 3.0,
  "tp_atr_mult": 3.0,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | adx:16, adx:18.00, adx:15min, atr:10, sl:3.00, tp:2.50, capi:0.80, marg:3.00 | 4 OOS trials. Best PnL: -111.02. PnLs: [-111, -223, -294...] |
| 1    | COMPLETED | adx:22, adx:15.00, adx:30min, atr:20, sl:3.00, tp:3.00, capi:0.80, marg:3.00 | 4 OOS trials. Best PnL: -273.38. PnLs: [-273, -304, -324...] |

