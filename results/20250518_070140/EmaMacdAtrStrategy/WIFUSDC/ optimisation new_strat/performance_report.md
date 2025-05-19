# WFO Performance Report: EmaMacdAtrStrategy - WIFUSDC ( optimisation new_strat)

## WFO Configuration
- Strategy: EmaMacdAtrStrategy
- Pair: WIFUSDC
- Context:  optimisation new_strat
- WFO Run Timestamp: 20250518_070140

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 2108.8300*
```json
{
  "ema_short_period": 13,
  "ema_long_period": 80,
  "indicateur_frequence_ema": "30min",
  "macd_fast_period": 7,
  "macd_slow_period": 27,
  "macd_signal_period": 8,
  "indicateur_frequence_macd": "30min",
  "atr_period_sl_tp": 14,
  "atr_base_frequency_sl_tp": "30min",
  "sl_atr_mult": 2.5,
  "tp_atr_mult": 2.0,
  "atr_volatility_filter_period": 11,
  "indicateur_frequence_atr_volatility": "1h",
  "atr_volatility_threshold_mult": 2.0,
  "taker_pressure_indicator_period": 16,
  "indicateur_frequence_taker_pressure": "5min",
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | ema:10, ema:75, ema:3min, macd:20, macd:25, macd:14, macd:15min, atr:19, sl:1.25, tp:2.00, atr:19, volatility:5min, atr:0.60, take:30, pressure:3min, capi:0.80, marg:3.00 | 1 OOS trials. Best PnL: 181.48. PnLs: [181...] |
| 1    | COMPLETED | ema:13, ema:80, ema:30min, macd:7, macd:27, macd:8, macd:30min, atr:14, sl:2.50, tp:2.00, atr:11, volatility:1h, atr:2.00, take:16, pressure:5min, capi:0.80, marg:3.00 | 10 OOS trials. Best PnL: 2108.83. PnLs: [2109, 783, 718...] |

