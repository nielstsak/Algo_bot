# WFO Performance Report: EmaMacdAtrStrategy - PEPEUSDC ( optimisation new_strat)

## WFO Configuration
- Strategy: EmaMacdAtrStrategy
- Pair: PEPEUSDC
- Context:  optimisation new_strat
- WFO Run Timestamp: 20250519_060729

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 239.7600*
```json
{
  "ema_short_period": 49,
  "ema_long_period": 55,
  "indicateur_frequence_ema": "5min",
  "macd_fast_period": 5,
  "macd_slow_period": 19,
  "macd_signal_period": 14,
  "indicateur_frequence_macd": "30min",
  "atr_period_sl_tp": 13,
  "atr_base_frequency_sl_tp": "30min",
  "sl_atr_mult": 2.5,
  "tp_atr_mult": 2.0,
  "atr_volatility_filter_period": 21,
  "indicateur_frequence_atr_volatility": "15min",
  "atr_volatility_threshold_mult": 1.8,
  "taker_pressure_indicator_period": 30,
  "indicateur_frequence_taker_pressure": "3min",
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | ema:43, ema:35, ema:15min, macd:15, macd:23, macd:10, macd:1h, atr:11, sl:1.75, tp:2.00, atr:17, volatility:1h, atr:0.90, take:23, pressure:5min, capi:0.80, marg:3.00 | 10 OOS trials. Best PnL: 103.57. PnLs: [104, -102, -10000...] |
| 1    | COMPLETED | ema:49, ema:55, ema:5min, macd:5, macd:19, macd:14, macd:30min, atr:13, sl:2.50, tp:2.00, atr:21, volatility:15min, atr:1.80, take:30, pressure:3min, capi:0.80, marg:3.00 | 3 OOS trials. Best PnL: 239.76. PnLs: [240, -10000, -10000...] |

