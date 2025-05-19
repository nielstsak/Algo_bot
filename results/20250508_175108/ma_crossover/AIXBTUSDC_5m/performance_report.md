# WFO Performance Report: ma_crossover - AIXBTUSDC_5m

## WFO Configuration

*Note: Detailed WFO settings were not found in the summary file.*

## Out-of-Sample (OOS) Performance Summary

- **Average OOS Sharpe Ratio:** 20.5635
- **Total OOS PnL (All Folds):** 7646.04 USDC
- **Total Folds Run:** 2
- **Successful OOS Folds:** 2
## Fold-by-Fold OOS Results

*(Displaying primary metric: Sharpe Ratio)*

| Fold | Status    | Best IS Params (kama_period, fast, slow) | OOS Metric Value |
| :--- | :-------- | :--------------------------------------- | :--------------- |
| 0    | COMPLETED | p=?, f=?, s=?                            | 20.5336          |
| 1    | COMPLETED | p=?, f=?, s=?                            | 20.5935          |

## Parameters Selected for Live Config

*Parameters selected from the last completed fold (Fold 1)*

```json
{
  "fast_ma_period": 5,
  "slow_ma_period": 17,
  "ma_type": "ema",
  "atr_period": 14,
  "sl_atr_multiplier": 1.8,
  "tp_atr_multiplier": 1.2
}
```
