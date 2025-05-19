# WFO Performance Report: BbandsVolumeRsiStrategy - WIFUSDC ( optimisation new_strat)

## WFO Configuration
- Strategy: BbandsVolumeRsiStrategy
- Pair: WIFUSDC
- Context:  optimisation new_strat
- WFO Run Timestamp: 20250518_070140

## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)
*Detailed OOS performance is shown per fold below.*

## Parameters Selected for Live Configuration
*Selected from Fold 1 based on best OOS 'Total Net PnL USDC': 2391.0400*
```json
{
  "bbands_period": 21,
  "bbands_std_dev": 1.75,
  "indicateur_frequence_bbands": "5min",
  "volume_ma_period": 12,
  "indicateur_frequence_volume": "5min",
  "rsi_period": 12,
  "indicateur_frequence_rsi": "15min",
  "rsi_buy_breakout_threshold": 71.0,
  "rsi_sell_breakout_threshold": 26.0,
  "atr_period_sl_tp": 10,
  "atr_base_frequency_sl_tp": "15min",
  "sl_atr_mult": 2.0,
  "tp_atr_mult": 1.75,
  "taker_pressure_indicator_period": 27,
  "indicateur_frequence_taker_pressure": "5min",
  "capital_allocation_pct": 0.8,
  "margin_leverage": 3.0
}
```

## Fold-by-Fold OOS Results

| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |
| :--- | :-------- | :--------------------------- | :----------------------- |
| 0    | COMPLETED | bban:30, bbands:15min, volu:29, volume:15min, rsi:18, rsi:1h, rsi:85.00, rsi:18.00, atr:18, sl:2.00, tp:1.75, take:13, pressure:5min, capi:0.80, marg:3.00 | 1 OOS trials. Best PnL: 1639.85. PnLs: [1640...] |
| 1    | COMPLETED | bban:21, bbands:5min, volu:12, volume:5min, rsi:12, rsi:15min, rsi:71.00, rsi:26.00, atr:10, sl:2.00, tp:1.75, take:27, pressure:5min, capi:0.80, marg:3.00 | 2 OOS trials. Best PnL: 2391.04. PnLs: [2391, 83...] |

