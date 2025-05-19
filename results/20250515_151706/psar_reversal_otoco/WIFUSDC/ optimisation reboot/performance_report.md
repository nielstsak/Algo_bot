# WFO Performance Report: psar_reversal_otoco - WIFUSDC ( optimisation reboot)

## WFO Configuration
- Strategy: psar_reversal_otoco
- Pair: WIFUSDC
- Context:  optimisation reboot
- WFO Run Timestamp: 20250515_151706

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 594.2500*
```json
{
  "psar_step": 0.01,
  "psar_max_step": 0.12000000000000001,
  "indicateur_frequence_psar": "5min",
  "atr_period": 10,
  "atr_base_frequency": "1h",
  "sl_atr_mult": 3.0,
  "tp_atr_mult": 3.0,
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | psar:0.03, psar:0.10, psar:15min, atr:15, sl:2.50, tp:1.00, capi:0.80, marg:3.00 | 3 OOS trials. Best PnL: 332.26. PnLs: [332, -15, -145...] |
| 1    | COMPLETED | psar:0.01, psar:0.12, psar:5min, atr:10, sl:3.00, tp:3.00, capi:0.80, marg:3.00 | 6 OOS trials. Best PnL: 594.25. PnLs: [594, 448, 242...] |

